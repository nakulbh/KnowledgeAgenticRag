"""LangGraph agentic RAG workflow implementation."""

from typing import Literal, Dict, Any
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from pydantic import BaseModel, Field
from ..retrieval.retriever import create_retriever_tool_for_rag


# Pydantic model for document grading
class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


# Prompts
GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def create_rag_workflow(
    model_name: str = "openai:gpt-4o-mini",
    temperature: float = 0,
    collection_name: str = "rag_documents",
    chroma_host: str = "localhost",
    chroma_port: int = 8000
):
    """Create the agentic RAG workflow.
    
    Args:
        model_name: Name of the chat model to use
        temperature: Temperature for the model
        collection_name: ChromaDB collection name
        chroma_host: ChromaDB host
        chroma_port: ChromaDB port
        
    Returns:
        Compiled LangGraph workflow
    """
    
    # Initialize models
    response_model = init_chat_model(model_name, temperature=temperature)
    grader_model = init_chat_model(model_name, temperature=0)
    
    # Create retriever tool
    retriever_tool = create_retriever_tool_for_rag(
        collection_name=collection_name,
        host=chroma_host,
        port=chroma_port,
        n_results=5
    )
    
    def generate_query_or_respond(state: MessagesState):
        """Call the model to generate a response based on the current state.
        Given the question, it will decide to retrieve using the retriever tool,
        or simply respond to the user.
        """
        response = (
            response_model
            .bind_tools([retriever_tool])
            .invoke(state["messages"])
        )
        return {"messages": [response]}
    
    def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question."""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = (
            grader_model
            .with_structured_output(GradeDocuments)
            .invoke([{"role": "user", "content": prompt}])
        )
        score = response.binary_score
        
        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"
    
    def rewrite_question(state: MessagesState):
        """Rewrite the original user question."""
        messages = state["messages"]
        question = messages[0].content
        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}
    
    def generate_answer(state: MessagesState):
        """Generate an answer."""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}
    
    # Build the workflow
    workflow = StateGraph(MessagesState)
    
    # Define the nodes
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    
    # Define the edges
    workflow.add_edge(START, "generate_query_or_respond")
    
    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    
    # Edges taken after the retrieve node is called
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    
    # Add memory for conversation history
    memory = MemorySaver()
    
    # Compile the workflow
    graph = workflow.compile(checkpointer=memory)
    
    return graph


def run_rag_query(
    graph,
    query: str,
    thread_id: str = "default_thread"
) -> str:
    """Run a single query through the RAG workflow.
    
    Args:
        graph: Compiled LangGraph workflow
        query: User query
        thread_id: Thread ID for conversation history
        
    Returns:
        Response from the RAG system
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the workflow
    result = graph.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config
    )
    
    # Extract the final response
    final_message = result["messages"][-1]
    return final_message.content


def stream_rag_response(
    graph,
    query: str,
    thread_id: str = "default_thread"
):
    """Stream response from the RAG workflow.
    
    Args:
        graph: Compiled LangGraph workflow
        query: User query
        thread_id: Thread ID for conversation history
        
    Yields:
        Streaming response chunks
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": query}]},
        config=config
    ):
        for node, update in chunk.items():
            if "messages" in update and update["messages"]:
                yield update["messages"][-1]
