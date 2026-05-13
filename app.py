import streamlit as st
from PIL import Image


st.title("insPLAD 이상탐지 시스템")

uploaded = st.file_uploader("이미지 업로드", type=["jpg","png","jpeg"])

with st.sidebar:
    st.header("평가 설정")
    compare_mode = st.toggle("비교 평가 모드", value=False)
    if compare_mode:
        st.caption("세 조건을 모두 실행하고 RAGAS 점수를 비교합니다.")
        st.markdown("""
        - **A** gpt-4o + RAG *(baseline)*  
        - **B** gpt-4o-mini, RAG 없음  
        - **C** gpt-4o-mini + RAG *(현재 시스템)*
        """)


if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="업로드된 이미지", width=300)

    if st.button("분석 시작"):
        from pipeline.classify import classify
        with st.spinner("분류 중..."):
            class_id, class_name = classify(image)
        st.success(f"클래스: {class_name}")

        with st.spinner("이상탐지 중..."):
            from pipeline.anomaly import run_anomaly
            score, is_anomaly, heatmap, overlay = run_anomaly(image, class_id)

        col1, col2, col3 = st.columns(3)
        col1.image(image,   caption="원본")
        col2.image(heatmap, caption="히트맵")
        col3.image(overlay, caption="오버레이")

        status = "이상" if is_anomaly else "정상"
        st.metric("Anomaly Score", f"{score:.4f}", delta=status)

        with st.spinner("RAG 검색 중..."):
            from pipeline.rag import search
            query = f"{class_name} {'anomaly defect inspection' if is_anomaly else 'normal inspection'}"
            docs = search(query, class_name)

        from pipeline.llm import generate_report

        if compare_mode:
            # 세 조건 모두 생성
            with st.spinner("리포트 생성 중 (3개 조건)..."):
                report_a = generate_report(class_name, score, is_anomaly, docs,        model="gpt-4o")
                report_b = generate_report(class_name, score, is_anomaly, context_docs=None, model="gpt-4o-mini")
                report_c = generate_report(class_name, score, is_anomaly, docs,        model="gpt-4o-mini")

            # ── 3. 평가 ─────────────────────────────────────
            with st.spinner("RAGAS 평가 중 (3개 조건)..."):
                from pipeline.evaluation import evaluate_report
                scores = evaluate_report(query, docs, report_a, report_b, report_c)

            # ── 4. 탭으로 나란히 비교 ────────────────────────
            st.markdown("### 비교 평가 결과")

            # 점수 요약 테이블 (탭 위에 한눈에)
            import pandas as pd
            df = pd.DataFrame({
                "조건":               ["A: gpt-4o+RAG",      "B: mini, no RAG",    "C: mini+RAG"],
                "faithfulness":       [scores["A_gpt4o_rag"].get("faithfulness","-"),  scores["B_mini_no_rag"].get("faithfulness","-"),  scores["C_mini_rag"].get("faithfulness","-")],
                "answer_relevancy":   [scores["A_gpt4o_rag"].get("answer_relevancy","-"), scores["B_mini_no_rag"].get("answer_relevancy","-"), scores["C_mini_rag"].get("answer_relevancy","-")],
                "context_precision":  [scores["A_gpt4o_rag"].get("context_precision","-"), "-", scores["C_mini_rag"].get("context_precision","-")],
                "context_recall":     [scores["A_gpt4o_rag"].get("context_recall","-"),    "-", scores["C_mini_rag"].get("context_recall","-")],
            }).set_index("조건")
            st.dataframe(df, use_container_width=True)

            # 탭별 리포트 전문
            tab_a, tab_b, tab_c = st.tabs([
                "A: gpt-4o + RAG",
                "B: mini, RAG 없음",
                "C: mini + RAG (현재 시스템)",
            ])
            with tab_a:
                st.json(scores["A_gpt4o_rag"])
                st.markdown(report_a)
            with tab_b:
                st.json(scores["B_mini_no_rag"])
                st.markdown(report_b)
            with tab_c:
                st.json(scores["C_mini_rag"])
                st.markdown(report_c)

            # 저장은 현재 시스템(C) 기준
            report, ragas = report_c, scores["C_mini_rag"]

        else:
            # 기존 단일 모드
            with st.spinner("리포트 생성 중..."):
                report = generate_report(class_name, score, is_anomaly, docs, model="gpt-4o-mini")
            st.markdown("### 검사 리포트")
            st.markdown(report)

            with st.spinner("품질 평가 중..."):
                from pipeline.evaluation import evaluate_single
                ragas = evaluate_single(query, docs, report)
            st.markdown("### RAGAS 평가 점수")
            st.json(ragas)

        # ── 5. 저장 ─────────────────────────────────────────
        with st.spinner("결과 저장 중..."):
            from pipeline.storage import save_result
            rid = save_result(class_name, score, is_anomaly, report,
                              image, heatmap, overlay, ragas)
        st.success(f"저장 완료! ID: {rid}")