import { ChatGroq } from "@langchain/groq";
import { PromptTemplate } from "@langchain/core/prompts"
import { StringOutputParser } from '@langchain/core/output_parsers'
import { retriever } from './utils/step_2_retriever.js';
import { combineDocuments } from './utils/step_2_combineDocuments.js';

const groqAPIKey = process.env.GROQ_API_KEY

const llm = new ChatGroq({
    groqAPIKey,
    model:"llama-3.3-70b-versatile"
})

const originalQuestion = 'what are the technical requriements for running Scrimba? \
I only have a very old laptop which is not that powerful.'

const standaloneTemplate = 'Turn the following user question into a standalone quesiton. \
question: {question} standalone question:'

const standalonePrompt = PromptTemplate.fromTemplate(standaloneTemplate)

const template = `Answer friendly to a given question, depending on the given context \
Only answer from the given context and never make up answers. Apologize if you do not know the answer, \
and advice the user to email help@scrimba.com
context: {context}
question: {question}
answer:
`

const prompt = PromptTemplate.fromTemplate(template)

const chain = standalonePrompt.pipe(llm)
            .pipe(new StringOutputParser())
            .pipe(retriever)
            .pipe(combineDocuments)
            // .pipe(prompt)   // making a problem

const response = await chain.invoke({
    question: originalQuestion
})

console.log(response)