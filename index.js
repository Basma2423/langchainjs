import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { readFile } from 'fs/promises'
import dotenv from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase'
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'

dotenv.config()

try {
  const text = await readFile('scrimba-info.txt', 'utf-8')  
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500
  })
  
  const output = await splitter.createDocuments([text])

  const sbURL = process.env.SUPABASE_URL
  const sbAPIKey = process.env.SUPABASE_API_KEY
  const hfAPIKey = process.env.HUGGING_FACE_API_KEY

  const client = createClient(sbURL, sbAPIKey)
  
  // pick an embedding model from here:
  // https://huggingface.co/models?pipeline_tag=feature-extraction&sort=trending
  
  await SupabaseVectorStore.fromDocuments(
    output, 
    new  HuggingFaceInferenceEmbeddings({
      apiKey: hfAPIKey,
      model: "intfloat/multilingual-e5-large" 
    }), 
    {
      client,
      tableName: 'documents',
    } 

  )

    console.log('Successfully processed and uploaded the documents to Supabase vector store')
    console.log(`${output.length} document chunks were created and stored`)

} catch (err) {
  console.log(err)
}