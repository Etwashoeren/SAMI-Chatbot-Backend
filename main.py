from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
import json
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 환경 변수 로드
load_dotenv()
api_key = "[OPENAI_API_KEY]"

# Flask 애플리케이션 초기화
app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=api_key)

with open("sami_prompt.txt", "r", encoding="utf-8") as file:
    sami_prompt = file.read()

# Hugging Face 임베딩 모델 로드
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB 초기화 (Persistent 모드)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(name="knowledge_base")

# 데이터 유형별 컬렉션 생성
scholarship_collection = chroma_client.get_or_create_collection(name="scholarship_knowledge")
haksa_collection = chroma_client.get_or_create_collection(name="haksa_knowledge")
major_collection = chroma_client.get_or_create_collection(name="major_knowledge")

def process_scholarship_data():
    with open("scholarship.json", "r", encoding="utf-8") as file:
        scholarship_data = json.load(file)

    documents = []
    for item in scholarship_data:
        documents.append(
            f"{item['name']} - 대상: {item['eligibility']} / 지원 금액: {item['amount']} / 신청 방법: {item['application']}")
    return documents

def process_haksa_data():
    with open("Haksa.json", "r", encoding="utf-8") as file:
        haksa_data = json.load(file)

    documents = []
    for item in haksa_data:
        text = f"{item['name']} - 학기: {item['semester']} / 대상: {item['eligibility']} / 기간: {item['period']} / 신청 방법: {item['application']}"
        documents.append(text)
        if item.get("application"):
            text += f" / 신청 방법: {item['application']}"
    return documents

def process_major_data():
    with open("majors.json", "r", encoding="utf-8") as file:
        major_data = json.load(file)

    documents = []

    # 전공과목과 교수 정보를 처리합니다.
    for major in major_data:
        major_name = major["name"]  # 전공과목 이름
        major_phone = major["phoneNumber"]  # 전공과목 전화번호
        major_page = major["page"]  # 전공과목 홈페이지

        # 전공과목 정보 추가
        documents.append(f"전공: {major_name} - 전화번호: {major_phone} / 홈페이지: {major_page}")

        # 교수 정보 처리
        for item in major["faculty"]:
            text = f"전공: {major_name} - 교수명: {item['name']} / 연구실: {item['lab']} / 전화번호: {item['phoneNumber']}"
            if item.get("email"):
                text += f" / 이메일: {item['email']}"
            documents.append(text)
    return documents

# JSON 데이터 로드 및 벡터화
def load_data_to_chroma():
    collections = {
        "scholarship": (scholarship_collection, process_scholarship_data()),
        "haksa": (haksa_collection, process_haksa_data()),
        "major": (major_collection, process_major_data())
    }

    for category, (collection, data) in collections.items():
        if collection.count() == 0:
            print(f"{category} 데이터 저장 중...")
            for idx, text in enumerate(data):
                embedding = embedding_model.encode(text).tolist()
                collection.add(ids=[f"{category}_{idx}"], embeddings=[embedding], metadatas=[{"text": text}])
            print(f"{category} 데이터 저장 완료!")

load_data_to_chroma()

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get("question")
    print(f"user_question : {user_question}")

    if not user_question:
        return jsonify({"error": "질문을 입력해 주세요."}), 400

    # 사용자 질문 벡터화 후 검색
    user_embedding = embedding_model.encode(user_question).tolist()
    counts = len(user_embedding)

    haksa_results = haksa_collection.query(query_embeddings=[user_embedding], n_results=counts)
    scholarship_results = scholarship_collection.query(query_embeddings=[user_embedding], n_results=counts)
    major_results = major_collection.query(query_embeddings=[user_embedding], n_results=counts)

    haksa_docs = [doc["text"] for doc in haksa_results["metadatas"][0] if "text" in doc]
    scholarship_docs = [doc["text"] for doc in scholarship_results["metadatas"][0] if "text" in doc]
    major_docs = [doc["text"] for doc in major_results["metadatas"][0] if "text" in doc]

    # 검색된 데이터 log
    print(f"검색된 데이터 개수: {len(haksa_results['metadatas'][0])}")

    print(f"검색된 데이터 개수: {len(scholarship_results['metadatas'][0])}")

    print(f"검색된 데이터 개수: {len(major_results['metadatas'][0])}")

    for idx, doc in enumerate(haksa_results["metadatas"][0]):
        print(f"검색 결과 {idx + 1}: {doc['text']}")

    for idx, doc in enumerate(scholarship_results["metadatas"][0]):
        print(f"검색 결과 {idx + 1}: {doc['text']}")

    for idx, doc in enumerate(major_results["metadatas"][0]):
        print(f"검색 결과 {idx + 1}: {doc['text']}")

    haksa = "\n\n".join(haksa_docs)
    scholarship = "\n\n".join(scholarship_docs)
    major = "\n\n".join(major_docs)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sami_prompt},
            {"role": "system", "content": f"다음은 학사 일정 관련 참고 정보입니다:\n\n{haksa}"},
            {"role": "system", "content": f"다음은 장학금 관련 참고 정보입니다:\n\n{scholarship}"},
            {"role": "system", "content": f"다음은 전공 관련 참고 정보입니다:\n\n{major}"},
            {"role": "user", "content": user_question}
        ]
    )

    answer = completion.choices[0].message.content
    print(f"answer : {answer}")
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)

