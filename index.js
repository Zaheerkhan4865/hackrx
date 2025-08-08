import express from 'express';
import axios from 'axios';
import * as dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { v4 as uuidv4 } from 'uuid';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { DocxLoader } from '@langchain/community/document_loaders/fs/docx';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { GoogleGenerativeAI } from '@google/generative-ai';

dotenv.config();
const app = express();
app.use(express.json());

const PORT = 8000;
const TEAM_TOKEN = process.env.TEAM_TOKEN;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;

const ai = new GoogleGenerativeAI(GEMINI_API_KEY);
const processedDocs = new Set(); // in-memory cache

// Create SHA256 hash of document URL
function hashURL(url) {
  return crypto.createHash('sha256').update(url).digest('hex');
}

// Generic document processor for PDF/DOCX
async function processDocument(url) {
  // Detect file extension
  let ext = path.extname(new URL(url).pathname).toLowerCase();
  if (!['.pdf', '.docx'].includes(ext)) {
    throw new Error('Unsupported document format. Only PDF and DOCX are allowed.');
  }

  const localPath = `./temp-${uuidv4()}${ext}`;

  // Download file
  const response = await axios.get(url, { responseType: 'stream' });
  const writer = fs.createWriteStream(localPath);
  await new Promise((resolve, reject) => {
    response.data.pipe(writer);
    writer.on('finish', resolve);
    writer.on('error', reject);
  });

  // Pick correct loader
  let loader;
  if (ext === '.pdf') {
    loader = new PDFLoader(localPath);
  } else if (ext === '.docx') {
    loader = new DocxLoader(localPath);
  }

  const rawDocs = await loader.load();

  // Split into chunks
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 800,
    chunkOverlap: 100,
  });
  const chunks = await splitter.splitDocuments(rawDocs);

  // Generate embeddings
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: GEMINI_API_KEY,
    model: 'text-embedding-004',
  });

  // Store in Pinecone
  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index(PINECONE_INDEX_NAME);

  await PineconeStore.fromDocuments(chunks, embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });

  // Clean up
  fs.unlinkSync(localPath);
}

// Retry wrapper for Pinecone query
async function queryWithRetry(pineconeIndex, queryVector, retries = 2) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await pineconeIndex.query({
        topK: 10,
        vector: queryVector,
        includeMetadata: true,
      });
    } catch (e) {
      if (attempt === retries) throw e;
      console.warn(`🔁 Retry ${attempt} failed. Retrying...`);
      await new Promise((r) => setTimeout(r, 500));
    }
  }
}

// Answer question using Gemini + Pinecone with proof
async function answerQuestion(question, embeddings, pineconeIndex) {
  try {
    const queryVector = await embeddings.embedQuery(question);
    const results = await queryWithRetry(pineconeIndex, queryVector);

    const contextChunks = results.matches
      .map((m, idx) => `Source ${idx + 1}:\n${m.metadata.text}`)
      .join('\n\n---\n\n');

    const prompt = `
You are an expert insurance policy analyst. Provide concise yet comprehensive answers with maximum numerical precision.

CRITICAL REQUIREMENTS:
- For yes/no questions: Start with "Yes," or "No," then provide essential explanation
- Include ALL key numbers: exact days, months, years, percentages, amounts
- State important plan variations (Plan A vs Plan B vs Plan C) with key differences only
- Include essential age limits, waiting periods, and main coverage conditions
- Quote main benefit amounts, caps, and limits with numbers
- Mention only critical exceptions and key conditions
- Keep responses concise-medium length (50 words max) but include all essential details
You are an expert insurance claims evaluator. Based on the provided context from policy documents, evaluate the insurance claim and make a decision.

Original Query: {input}
Parsed Information: {parsedInfo}

Context from Policy Documents:
{context}

Instructions:
1. Analyze the query against the provided policy context
2. Make a clear decision: "Approved" or "Rejected"
3. If approved, determine the coverage amount (use null if not specified)
4. Provide detailed justification referencing specific clauses or sections
Question: ${question}

Context:
${contextChunks}
    `.trim();

    const model = ai.getGenerativeModel({ model: 'gemini-2.5-pro' });
    const result = await model.generateContent(prompt);
    const response = await result.response;
    return response.text().trim();
  } catch (error) {
    console.error(`❌ Gemini error for "${question}":`, error.message || error);
    return "An error occurred while processing this question.";
  }
}

// Main API route
app.post('/hackrx/run', async (req, res) => {
  try {
    const auth = req.headers.authorization;
    if (!auth || auth !== `Bearer ${TEAM_TOKEN}`) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const { documents, questions } = req.body;
    if (!documents || !Array.isArray(questions)) {
      return res.status(400).json({ error: 'Missing "documents" or "questions"' });
    }

    const hash = hashURL(documents);

    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: GEMINI_API_KEY,
      model: 'text-embedding-004',
    });

    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(PINECONE_INDEX_NAME);

    if (!processedDocs.has(hash)) {
      console.log("📄 Processing new document...");
      await processDocument(documents);
      processedDocs.add(hash);
      console.log("✅ Document processed and indexed.");
    } else {
      console.log("⏩ Using cached document.");
    }

    const answers = await Promise.all(
      questions.map((q) => answerQuestion(q, embeddings, pineconeIndex))
    );

    res.json({ answers });
  } catch (err) {
    console.error('❌ API error:', err.message || err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 API running at http://localhost:${PORT}/hackrx/run`);
});
