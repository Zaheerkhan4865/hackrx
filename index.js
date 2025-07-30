import express from "express";
import * as dotenv from "dotenv";
dotenv.config();

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

const ai = new GoogleGenAI({});
const History = [];

async function transformQuery(question) {
  History.push({
    role: "user",
    parts: [{ text: question }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Rephrase the "Follow Up user Question" into a standalone question that makes sense without the chat history. Only output the rewritten question.`,
    },
  });

  History.pop();
  return response.response.text();
}

async function generateAnswer(question) {
  const standaloneQuestion = await transformQuery(question);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: "text-embedding-004",
  });

  const queryVector = await embeddings.embedQuery(standaloneQuestion);

  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  const searchResults = await pineconeIndex.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
  });

  const context = searchResults.matches
    .map((match) => match.metadata.text)
    .join("\n\n---\n\n");

  History.push({
    role: "user",
    parts: [{ text: standaloneQuestion }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are an insurance advisor specialized in explaining health insurance policies in simple and clear terms.
Use ONLY the context provided to answer the question. If the answer is not in the context, say "I could not find the answer in the provided document."

Context: ${context}`,
    },
  });

  const answer = response.response.text();
  History.push({
    role: "model",
    parts: [{ text: answer }],
  });

  return answer;
}

app.post("/hackrx/run", async (req, res) => {
  try {
    const { question } = req.body;
    if (!question) return res.status(400).json({ error: "Question is required" });

    const answer = await generateAnswer(question);
    res.json({ answer });
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.get("/", (req, res) => {
  res.send("API is running. Use POST /hackrx/run");
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
