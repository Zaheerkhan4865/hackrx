import express from 'express';
import dotenv from 'dotenv';
import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { GoogleGenAI } from '@google/genai';

dotenv.config();
const app = express();
const PORT = process.env.PORT || 3000;
app.use(express.json());

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.post('/hackrx/run', async (req, res) => {
  try {
    const { documents, questions } = req.body;
    if (!documents || !questions) {
      return res.status(400).json({ error: 'Missing documents or questions' });
    }

    // Step 1: Download PDF from URL
    const pdfPath = path.join(__dirname, 'temp.pdf');
    const fileResponse = await fetch(documents);
    const fileStream = fs.createWriteStream(pdfPath);
    await new Promise((resolve, reject) => {
      fileResponse.body.pipe(fileStream);
      fileResponse.body.on('error', reject);
      fileStream.on('finish', resolve);
    });

    // Step 2: Load and chunk the document
    const loader = new PDFLoader(pdfPath);
    const rawDocs = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const docs = await splitter.splitDocuments(rawDocs);

    // Step 3: Embed and store vectors in Pinecone
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });

    const pinecone = new Pinecone();
    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    await PineconeStore.fromDocuments(docs, embeddings, {
      pineconeIndex: index,
      maxConcurrency: 5,
    });

    // Step 4: Answer each question
    const responseData = {};

    for (const question of questions) {
      const vector = await embeddings.embedQuery(question);
      const results = await index.query({
        vector,
        topK: 10,
        includeMetadata: true,
      });

      const context = results.matches
        .map(match => match.metadata.text)
        .join('\n\n---\n\n');

      const response = await ai.models.generateContent({
        model: 'gemini-2.0-pro',
        contents: [
          {
            role: 'user',
            parts: [{ text: `Context:\n${context}\n\nQuestion:\n${question}` }],
          },
        ],
        config: {
          systemInstruction: `You are an insurance advisor answering only using the given context. If the answer is not in the document, reply: "I could not find the answer in the provided document."`,
        },
      });

      responseData[question] = response.text;
    }

    res.status(200).json({ answers: responseData });
  } catch (err) {
    console.error('Error in /hackrx/run:', err);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

app.get('/', (req, res) => {
  res.send('Health Insurance QA API is running!');
});

app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
