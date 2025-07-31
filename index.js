import express from 'express';
import axios from 'axios';
import * as dotenv from 'dotenv';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { GoogleGenAI } from '@google/genai';

dotenv.config();
const app = express();
app.use(express.json());

const PORT = 8000;
const ai = new GoogleGenAI({});
const TEAM_TOKEN = "185e9c4657f138c2ed3e69c3a85da7d3ae371a2ca037dc5ab517d544f3256ec0";

app.post('/hackrx/run', async (req, res) => {
  try {
    // Token Validation
    const auth = req.headers.authorization;
    if (!auth || auth !== `Bearer ${TEAM_TOKEN}`) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    const { documents, questions } = req.body;
    if (!documents || !Array.isArray(questions)) {
      return res.status(400).json({ error: "Missing documents or questions array" });
    }

    // Download PDF
    const tempFilePath = `./temp-${uuidv4()}.pdf`;
    const response = await axios.get(documents, { responseType: 'stream' });
    const writer = fs.createWriteStream(tempFilePath);
    await new Promise((resolve, reject) => {
      response.data.pipe(writer);
      writer.on('finish', resolve);
      writer.on('error', reject);
    });

    // Load + Split + Embed + Store
    const pdfLoader = new PDFLoader(tempFilePath);
    const rawDocs = await pdfLoader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunkedDocs = await splitter.splitDocuments(rawDocs);

    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });

    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
      pineconeIndex,
      maxConcurrency: 5,
    });

    fs.unlinkSync(tempFilePath); // Cleanup

    // Answer each question
    const answers = [];
    for (let question of questions) {
      const queryVector = await embeddings.embedQuery(question);
      const results = await pineconeIndex.query({
        topK: 10,
        vector: queryVector,
        includeMetadata: true,
      });

      const context = results.matches
        .map(m => m.metadata.text)
        .join('\n\n---\n\n');

      const geminiResponse = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: question }] }],
        config: {
          systemInstruction: `You are an insurance advisor specialized in health insurance policies.
Answer using only this context. If not found, reply: "I could not find the answer in the provided document."

Context:
${context}
          `,
        },
      });

      answers.push(geminiResponse.text || "I could not find the answer in the provided document.");
    }

    res.json({ answers });
  } catch (error) {
    console.error("❌ Error:", error);
    res.status(500).json({ error: "Something went wrong" });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 API running at http://localhost:${PORT}/hackrx/run`);
});
