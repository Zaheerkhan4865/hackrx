// index.js
import express from 'express';
import * as dotenv from 'dotenv';
dotenv.config();

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

const app = express();
app.use(express.json());

const ai = new GoogleGenAI({});
const History = [];

async function transformQuery(question) {
  History.push({
    role: 'user',
    parts: [{ text: question }]
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the chat history, rephrase the "Follow Up user Question" into a complete, standalone question.
Only output the rewritten question and nothing else.`
    }
  });

  History.pop();
  return response.text;
}

async function chatting(question) {
  const queries = await transformQuery(question);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
  });

  const queryVector = await embeddings.embedQuery(queries);

  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  const searchResults = await pineconeIndex.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
  });

  const context = searchResults.matches
    .map(match => match.metadata.text)
    .join("\n\n---\n\n");

  History.push({
    role: 'user',
    parts: [{ text: queries }]
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are an insurance advisor specialized in explaining health insurance policies clearly.
Use the context below to answer the user's question.
If not found in the context, say: "I could not find the answer in the provided document."

Context: ${context}`
    }
  });

  History.push({
    role: 'model',
    parts: [{ text: response.text }]
  });

  return response.text;
}

// API Endpoint for /hackrx/run
app.post('/hackrx/run', async (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ error: "Missing 'question' field in request body." });
  }

  try {
    const answer = await chatting(question);
    return res.json({ answer });
  } catch (err) {
    console.error("Error in /hackrx/run:", err);
    return res.status(500).json({ error: "Internal Server Error" });
  }
});

app.get('/', (req, res) => {
  res.send("🚀 HackRx Insurance Agent API is live.");
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});
