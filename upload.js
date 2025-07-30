import express from 'express';
import multer from 'multer';
import * as dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

dotenv.config();

const app = express();
const PORT = 3000;

// Multer setup for handling file uploads
const upload = multer({ dest: 'uploads/' });

// GET route to serve the upload HTML form
app.get('/upload', (req, res) => {
  res.send(`
    <h2>Upload a PDF</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".pdf" required />
      <button type="submit">Upload</button>
    </form>
  `);
});

// POST route to handle PDF upload and indexing
app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    const filePath = req.file.path;

    // Load the PDF
    const pdfLoader = new PDFLoader(filePath);
    const rawDocs = await pdfLoader.load();
    console.log("✅ PDF loaded");

    // Split into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log("✅ Chunking complete");

    // Create embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });
    console.log("✅ Embedding model ready");

    // Pinecone connection
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    console.log("✅ Pinecone connected");

    // Store documents in Pinecone
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
      pineconeIndex,
      maxConcurrency: 5,
    });

    console.log("✅ Document indexed successfully");

    // Delete uploaded file after processing
    fs.unlinkSync(filePath);

    res.send('✅ Document uploaded and indexed successfully.');

  } catch (error) {
    console.error("❌ Error processing document:", error);
    res.status(500).send('Something went wrong while uploading.');
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
