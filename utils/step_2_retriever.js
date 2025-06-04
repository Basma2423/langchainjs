import { createClient } from '@supabase/supabase-js'
import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase'
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'
import dotenv from 'dotenv'

dotenv.config()
const hfAPIKey = process.env.HUGGING_FACE_API_KEY
const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: hfAPIKey,
    model: "intfloat/multilingual-e5-large" 
}) 

const sbURL = process.env.SUPABASE_URL
const sbAPIKey = process.env.SUPABASE_API_KEY
const client = createClient(sbURL, sbAPIKey)

const vectorStore = new SupabaseVectorStore(embeddings, {
    client,
    tableName: 'documents',
    queryName: 'match_documents'
})

const retriever = vectorStore.asRetriever()

export {retriever}