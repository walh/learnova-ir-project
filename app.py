
import streamlit as st
import pandas as pd
import re
import nltk
import tempfile
import html
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from urllib.parse import urlparse
import pyterrier as pt

st.set_page_config(
    page_title="Learnova | Find your next skill",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #f5f8ff;
    color: #142344;
}
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(120deg, #09265c, #0056d2);
    padding: 38px;
    border-radius: 24px;
    color: white;
    margin-bottom: 20px;
}
.hero h1 {
    color: white;
    font-size: 48px;
    letter-spacing: -2px;
    margin: 8px 0;
}
.hero p {
    color: #dceaff;
    font-size: 18px;
}
.eyebrow {
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
    color: #b8d6ff;
}
.course {
    background: white;
    border: 1px solid #dce5f5;
    border-radius: 16px;
    padding: 20px;
    min-height: 190px;
}
.course h3 {
    color: #142344;
    margin: 8px 0;
}
.meta {
    color: #526481;
    font-size: 14px;
}
.tag {
    display: inline-block;
    background: #e9f1ff;
    color: #164c9e;
    padding: 4px 9px;
    border-radius: 8px;
    font-size: 12px;
    margin: 3px;
}
.stButton button {
    border-radius: 10px;
}
[data-testid="stMetric"] {
    background: white;
    border-radius: 14px;
    padding: 16px;
    border: 1px solid #dce5f5;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing the course search index...")
def setup():
    data = pd.read_csv(
        "cleaned_dataset_restored.csv",
        dtype=str,
        keep_default_na=False
    )

    coll = pd.read_csv(
        "collection_restored.csv",
        dtype=str,
        keep_default_na=False
    )

    def normalize(value):
        return " ".join(str(value).split()).casefold()

    if (
        len(data) != len(coll)
        or data["Course Name"].map(normalize).tolist()
        != coll["title"].map(normalize).tolist()
    ):
        st.error("Dataset alignment failed.")
        st.write("Dataset rows:", len(data))
        st.write("Collection rows:", len(coll))

        differences = [
            (i, a, b)
            for i, (a, b) in enumerate(
                zip(data["Course Name"], coll["title"])
            )
            if normalize(a) != normalize(b)
        ]

        st.write("First differences:", differences[:5])
        st.stop()

    data["docno"] = coll["docno"]

    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        sw = set(stopwords.words("english"))

    stem = PorterStemmer()

    def prep(s):
        s = re.sub(
            r"https?://\S+|www\.\S+",
            "",
            str(s).lower()
        )
        s = re.sub(r"[^\w\s]", " ", s)

        return " ".join(
            stem.stem(w)
            for w in s.split()
            if w not in sw
        )

    docs = [
        {
            "docno": r.docno,
            "text": prep(r.title + " " + r.text)
        }
        for r in coll.itertuples()
    ]

    if not pt.java.started():
        pt.java.init()

    idx = pt.terrier.IterDictIndexer(
        tempfile.mkdtemp(),
        stemmer=None,
        stopwords=None
    ).index(docs)

    return data.set_index("docno"), idx, prep


data, idx, prep = setup()

if "saved" not in st.session_state:
    st.session_state.saved = []

if "query" not in st.session_state:
    st.session_state.query = ""


@st.cache_data(show_spinner=False)
def search(q, model):
    processed = prep(q)

    if not processed:
        return pd.DataFrame(
            columns=["docno", "score", "rank"]
        )

    retriever = pt.terrier.Retriever(
        idx,
        wmodel=model,
        num_results=10,
        controls={"applypipeline": "off"}
    )

    return retriever.transform(
        pd.DataFrame([
            {"qid": "1", "query": processed}
        ])
    )


def esc(x):
    return html.escape(str(x))


def course(doc, rank=None, score=None, key=""):
    r = data.loc[doc]

    title = str(r["Course Name"]).title()
    rating = r["Course Rating"]
    level = str(r["Difficulty Level"]).title()
    url = str(r["Course URL"])
    skills = str(r["Skills"])
    university = str(r["University"]).title()

    rank_text = f"#{rank} · " if rank else ""

    st.markdown(
        f"""
        <div class="course">
            <div class="meta">
                {rank_text}{esc(university)}
            </div>
            <h3>{esc(title)}</h3>
            <span class="tag">★ {esc(rating)}</span>
            <span class="tag">{esc(level)}</span>
            <p class="meta">{esc(skills[:160])}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if score is not None:
        st.caption(
            f"Retrieval score: {score:.4f} · {doc}"
        )

    a, b = st.columns([3, 1])
    parsed = urlparse(url)

    if (
        parsed.scheme == "https"
        and parsed.hostname in (
            "www.coursera.org",
            "coursera.org"
        )
    ):
        a.link_button(
            "View on Coursera ↗",
            url,
            use_container_width=True
        )

    label = (
        "Saved ✓"
        if doc in st.session_state.saved
        else "Save ♡"
    )

    if b.button(label, key=f"{key}-{doc}"):
        if doc in st.session_state.saved:
            st.session_state.saved.remove(doc)
        else:
            st.session_state.saved.append(doc)

        st.rerun()

    with st.expander("Course details"):
        st.write(r["Course Description"])


# HEADER

header, link = st.columns([5, 1])

header.markdown("### ◈ Learnova")

link.link_button(
    "Coursera ↗",
    "https://www.coursera.org/"
)


# HERO SECTION

st.markdown("""
<div class="hero">
    <div class="eyebrow">
        YOUR NEXT CHAPTER STARTS HERE
    </div>
    <h1>
        Find the skill.<br>
        Unlock your potential.
    </h1>
    <p>
        Discover courses that match what you want
        to learn, powered by information retrieval.
    </p>
</div>
""", unsafe_allow_html=True)


# STATISTICS

a, b, c = st.columns(3)

a.metric("Courses to discover", f"{len(data):,}")
b.metric("Learning providers", data.University.nunique())
c.metric("Search engines", 2)


# NAVIGATION

t1, t2, t3 = st.tabs([
    "Explore & search",
    "Compare models",
    "My saved courses & insights"
])


# SEARCH PAGE

with t1:
    st.markdown("#### What do you want to learn?")

    chips = st.columns(5)

    suggestions = [
        "Python",
        "Machine learning",
        "Cybersecurity",
        "Digital marketing",
        "Public speaking"
    ]

    for col, suggestion in zip(chips, suggestions):
        col.button(
            suggestion,
            on_click=lambda v=suggestion:
                st.session_state.update(query=v),
            use_container_width=True
        )

    q = st.text_input(
        "Search courses",
        key="query",
        placeholder="Try: I want to learn data analysis..."
    )

    model = st.radio(
        "Retrieval model",
        ["TF_IDF", "BM25"],
        format_func=lambda x: x.replace("_", "-"),
        horizontal=True
    )

    if q.strip():
        res = search(q, model)

        st.caption(
            f'{len(res)} ranked results · '
            f'{model.replace("_", "-")} · '
            f'Query terms: {prep(q)}'
        )

        if res.empty:
            st.info(
                "No matching courses. Try a different search."
            )

        cols = st.columns(2)

        for i, r in enumerate(res.itertuples()):
            with cols[i % 2]:
                course(
                    r.docno,
                    i + 1,
                    r.score,
                    "search"
                )

    else:
        st.info(
            "Choose a suggested search or enter your own learning goal."
        )


# MODEL COMPARISON

with t2:
    st.subheader("Compare retrieval models")

    cq = st.text_input(
        "Compare a query",
        value="public speaking and presentation skills"
    )

    left, right = st.columns(2)

    if cq.strip():
        for col, m in [
            (left, "TF_IDF"),
            (right, "BM25")
        ]:
            with col:
                st.subheader(m.replace("_", "-"))

                rr = search(cq, m)

                for i, r in enumerate(rr.itertuples()):
                    title = str(
                        data.loc[r.docno, "Course Name"]
                    ).title()

                    st.write(f"{i + 1}. {title}")
                    st.caption(f"Score: {r.score:.4f}")

        st.caption(
            "Raw scores use different scales; "
            "compare rankings, not score magnitudes."
        )


# SAVED COURSES AND INSIGHTS

with t3:
    st.subheader("Your learning list")

    if not st.session_state.saved:
        st.info(
            "Save courses to build your learning list."
        )

    for d in list(st.session_state.saved):
        course(d, key="saved")

    st.divider()

    st.subheader("Behind the search")

    st.write(
        "Learnova normalizes, removes stopwords "
        "and stems text before indexing. "
        "TF-IDF and BM25 rank matching courses."
    )

    st.dataframe(
        pd.DataFrame({
            "Metric": [
                "MAP",
                "Precision@10",
                "Recall@10"
            ],
            "TF-IDF": [0.587, 0.865, 0.629],
            "BM25": [0.580, 0.850, 0.621]
        }),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Reported notebook evaluation on 20 queries; "
        "not live user-search accuracy."
    )


# FOOTER

st.divider()

st.caption(
    "Learnova · DS321 academic project · "
    "Course metadata from a Coursera dataset. "
    "Independent project, not affiliated with Coursera. "
    "Course availability and ratings may change."
)
