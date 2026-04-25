from fastapi import FastAPI
from api.schemas import QuestionRequest, AnswerResponse
from api.services import answer_question

app = FastAPI(title="AI Learning Assistant API")


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    answer = answer_question(request.question)
    return {"answer": answer}