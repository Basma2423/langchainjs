import { ChatGroq } from "@langchain/groq";
import { PromptTemplate } from "@langchain/core/prompts"
import { StringOutputParser } from '@langchain/core/output_parsers'
import { retriever } from './utils/step_2_retriever.js';
import { combineDocuments } from './utils/step_2_combineDocuments.js'
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";

const groqAPIKey = process.env.GROQ_API_KEY

const llm = new ChatGroq({
    groqAPIKey,
    model:"llama-3.3-70b-versatile"
})

const standaloneQuestionTemplate = 'Given a question, convert it to a standalone question. question: {question} standalone question:'
const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate)

const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided. Try to find the answer in the context. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@scrimba.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
question: {question}
answer: `
const answerPrompt = PromptTemplate.fromTemplate(answerTemplate)

const standaloneChain = RunnableSequence.from([
    standaloneQuestionPrompt,
    llm,
    new StringOutputParser()
])

const contextChain = RunnableSequence.from([
    prevResult => prevResult.standalone_question,
    retriever,
    combineDocuments
])

const answerChain = RunnableSequence.from([
    answerPrompt,
    llm,
    new StringOutputParser()
])

const chain = RunnableSequence.from([
    {
        standalone_question: standaloneChain,
        original_input: new RunnablePassthrough()
    },
    {
        context: contextChain,
        question: ({original_input}) => original_input.question
    },
    answerChain
])

const response = await chain.invoke({
    question: 'What are the technical requirements for running Scrimba? I only have a very old laptop which is not that powerful.'
})

console.log(response)