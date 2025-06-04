import dotenv from 'dotenv'
import { ChatGroq } from "@langchain/groq";
import { PromptTemplate } from "@langchain/core/prompts"

dotenv.config()

const groqAPIKey = process.env.GROQ_API_KEY

const llm = new ChatGroq({
    groqAPIKey,
    model:"llama-3.3-70b-versatile"
})

const template = 'Turn the following user question into a standalone quesiton. \
question: {question} standalone question:'

const  prompt = PromptTemplate.fromTemplate(template)

const chain = prompt.pipe(llm)

const response = await chain.invoke({
    question: 'what are the technical requriements for running Scrimba? \
    I only have a very old laptop which is not that powerful.'
})

console.log(response)