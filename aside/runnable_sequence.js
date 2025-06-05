import { ChatGroq } from "@langchain/groq";
import { PromptTemplate } from "@langchain/core/prompts"
import { StringOutputParser } from '@langchain/core/output_parsers'
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import dotenv from 'dotenv'
dotenv.config()

const groqAPIKey = process.env.GROQ_API_KEY

const llm = new ChatGroq({
    groqAPIKey,
    model:"llama-3.3-70b-versatile"
})

const input = "i dont liked mondays"

// templates
const punctTemplate = ` Given a sentence, add punctuation where needed.
    sentence: {sentence}
    sentence with punctuation: 
`
const grammerTemplate = ` Given a sentence, correct the grammer.
    sentence: {punctuated_sentence}
    corrected sentence: 
`
const translateTemplate = ` Given a sentecence, translate it to {language}.
    sentence: {corrected_sentence}
    translated sentence: 
` 

// prompts
const punctPrompt= PromptTemplate.fromTemplate(punctTemplate) 
const grammerPrompt= PromptTemplate.fromTemplate(grammerTemplate) 
const translatePrompt= PromptTemplate.fromTemplate(translateTemplate) 

// chains
const punctChain = RunnableSequence.from([
    punctPrompt,
    llm,
    new StringOutputParser()
])

const grammerChain = RunnableSequence.from([
    grammerPrompt,
    llm,
    new StringOutputParser()
])

const translateChain = RunnableSequence.from([
    translatePrompt,
    llm,
    new StringOutputParser()
])

const chain = RunnableSequence.from([
   {//1.
        punctuated_sentence: punctChain,
        original_input: new RunnablePassthrough(),
   },
   {//2.
        corrected_sentence: grammerChain,
        language: ({original_input}) => original_input.language
   },
    //3.
    translateChain
])

const response = await chain.invoke({
    sentence: input,
    language: 'french'
})

console.log(response)