#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.7

# @Time   : 2026/2/21 11:16
# @Author : 'Lou Zehua'
# @File   : rag_demo.py 

import os
import sys

from langchain.schema import BaseRetriever

from etc import filePathConf
from script.build_dataset.repo_filter import get_filenames_by_repo_names

if '__file__' not in globals():
    # !pip install ipynbname  # Remove comment symbols to solve the ModuleNotFoundError
    import ipynbname

    nb_path = ipynbname.path()
    __file__ = str(nb_path)
cur_dir = os.path.dirname(__file__)
pkg_rootdir = os.path.dirname(os.path.dirname(cur_dir))  # os.path.dirname()向上一级，注意要对应工程root路径
if pkg_rootdir not in sys.path:  # 解决ipynb引用上层路径中的模块时的ModuleNotFoundError问题
    sys.path.append(pkg_rootdir)
    print('-- Add root directory "{}" to system path.'.format(pkg_rootdir))

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

from script.rag_demo.data_preprocess import load_and_transform_csv
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
from langchain.schema import Document


def create_filtered_retriever(vectorstore, base_retriever, exclude_patterns=None):
    """创建过滤检索器的工厂函数"""

    exclude_patterns = exclude_patterns or ["GitHub_Service_External_Links"]

    class _FilteredRetriever(BaseRetriever):
        def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            # 获取文档
            docs = base_retriever.get_relevant_documents(query)

            # 过滤
            filtered = []
            for doc in docs:
                if not any(p in doc.page_content for p in exclude_patterns):
                    filtered.append(doc)

            # 如果过滤后太少，补充检索
            if len(filtered) < 3:
                repo_docs = vectorstore.similarity_search(
                    query + " Repo reference",
                    k=10
                )
                for doc in repo_docs:
                    if not any(p in doc.page_content for p in exclude_patterns):
                        if doc not in filtered:
                            filtered.append(doc)
                            if len(filtered) >= 5:
                                break

            return filtered[:5]

    return _FilteredRetriever()


def ask_question(question, qa_chain):
    result = qa_chain.invoke({"query": question})
    print("Answer:")
    print(result["result"])
    print("\nSource Documents (Provenance):")
    for i, doc in enumerate(result["source_documents"]):
        print(f"Source {i+1}: Repo:[ID:{doc.metadata['tar_repo_id']}]{doc.metadata['tar_repo_name']}: {doc.page_content[:200]}...")


if __name__ == '__main__':
    year = 2023
    i = 0
    # 1. 加载并切分文档
    repo_names = ["apache/kylin", "tikv/tikv"]
    filenames = get_filenames_by_repo_names(repo_names, year)
    demo_path = os.path.abspath(
        os.path.join(filePathConf.absPathDict[filePathConf.DBMS_REPOS_GH_CORE_DIR], filenames[i]))
    docs = load_and_transform_csv(demo_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    os.environ["DASHSCOPE_API_KEY"] = "sk-xxx"  # 替换为阿里云 Key
    os.environ["ECNU_API_KEY"] = "sk-xxx"
    os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-xxx"

    # 2. 创建向量库 (持久化到本地，名为 .chroma_db)
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2"  # 也可以使用 "text-embedding-v1" 或 "text-embedding-v3"
        # dimensions=1024 # 如果使用 v3 版本，可以指定维度，v2 不需要
    )

    persist_dir = os.path.join(filePathConf.absPathDict[filePathConf.DATA_DIR], ".chroma_db")
    # FAISS 不需要 persist_directory参数，可以手动保存
    print("Vectorstore created successfully!")
    if not os.path.exists(os.path.join(persist_dir, "index.faiss")):
        print("首次运行，创建向量数据库...")
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )
        vectorstore.save_local(persist_dir)
    else:
        print("检测到已有数据库，直接加载...")
        vectorstore = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # 3. 构建检索器
    # 创建基础检索器
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 15,  # 返回文档数
            "fetch_k": 50,  # 获取候选数
            "lambda_mult": 0.7  # 平衡相关性和多样性
        }
    )
    # 包装成过滤检索器
    filtered_retriever = create_filtered_retriever(
        vectorstore=vectorstore,
        base_retriever=base_retriever,
        exclude_patterns=["GitHub_Service_External_Links"]
    )

    # 4. 初始化大模型
    llm = ChatOpenAI(
        api_key=os.environ.get("ECNU_API_KEY"),
        base_url="https://chat.ecnu.edu.cn/open/api/v1",
        model="ecnu-turbo",
        max_tokens=2048,  # <--- 将这里修改为一个较小的值，例如 2048
    )

    # llm = ChatOpenAI(
    #     api_key=os.environ.get("OPENROUTER_API_KEY"),
    #     base_url="https://openrouter.ai/api/v1",
    #     model="qwen/qwen3.5-plus-02-15",
    #     max_tokens=15000,  # <--- 将这里修改为一个较小的值，例如 2048
    # )

    # 5. 组装 RAG 链
    # stuff 代表把所有检索到的文档片段都塞进 prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=filtered_retriever,
        return_source_documents=True  # 关键：返回来源，用于展示“引用了哪条数据”
    )

    # 示例问题（模拟 OAG 场景）
    ask_question(f"{repo_names[i].replace('/', ' ')} 项目引用的实体相关的Repo有哪些？", qa_chain=qa_chain)
