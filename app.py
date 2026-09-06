import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# FILE PATH
# ============================================================

BASE_DIR = Path(__file__).parent


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_project_files():

    regression_model = joblib.load(
        BASE_DIR / "best_regression_model.pkl"
    )

    classification_model = joblib.load(
        BASE_DIR / "best_classification_model.pkl"
    )

    recommendation_df = joblib.load(
        BASE_DIR / "recommendation_data.pkl"
    )

    similarity_matrix = joblib.load(
        BASE_DIR / "similarity_matrix.pkl"
    )

    return (
        regression_model,
        classification_model,
        recommendation_df,
        similarity_matrix
    )


try:

    (
        regression_model,
        classification_model,
        recommendation_df,
        similarity_matrix
    ) = load_project_files()

except Exception as e:

    st.error("Error loading project files.")

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌍 Tourism Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📈 Rating Prediction",
        "🔍 Visit Mode Prediction",
        "⭐ Recommendations"
    ]
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("🌍 Tourism Experience Analytics")

    st.write(
        """
        A Machine Learning project for analyzing tourism
        experiences, predicting ratings and visit modes,
        and recommending similar attractions.
        """
    )

    st.divider()

    st.subheader("Project Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info("📈 Rating Prediction")

        st.write(
            "Predict attraction ratings using "
            "Random Forest Regression."
        )

    with col2:

        st.info("🔍 Visit Mode Prediction")

        st.write(
            "Predict visit modes using "
            "Random Forest Classification."
        )

    with col3:

        st.info("⭐ Recommendations")

        st.write(
            "Find similar attractions using "
            "TF-IDF and Cosine Similarity."
        )

    st.divider()

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Best Regression R²",
            "0.1109"
        )

    with col2:

        st.metric(
            "Classification Accuracy",
            "52.78%"
        )

    with col3:

        st.metric(
            "Highest Similarity",
            "0.8747"
        )


# ============================================================
# RATING PREDICTION PAGE
# ============================================================

elif page == "📈 Rating Prediction":

    st.title("📈 Rating Prediction")

    st.write(
        "Enter tourism information to predict the rating."
    )

    st.divider()


    # Get exact features used during training
    if hasattr(regression_model, "feature_names_in_"):

        regression_features = list(
            regression_model.feature_names_in_
        )

    else:

        regression_features = [
            "UserId",
            "VisitYear",
            "VisitMonthNum",
            "AttractionId",
            "ContinentId",
            "RegionId",
            "CountryId",
            "CityId",
            "AttractionCityId"
        ]


    st.info(
        f"Regression model expects "
        f"{len(regression_features)} features."
    )


    input_values = {}


    col1, col2 = st.columns(2)


    for i, feature in enumerate(regression_features):

        target_column = col1 if i % 2 == 0 else col2

        with target_column:

            if feature == "UserId":

                input_values[feature] = st.number_input(
                    "User ID",
                    min_value=0,
                    value=1000,
                    key=f"reg_{feature}"
                )


            elif feature == "VisitYear":

                input_values[feature] = st.number_input(
                    "Visit Year",
                    min_value=2000,
                    max_value=2030,
                    value=2025,
                    key=f"reg_{feature}"
                )


            elif feature in [
                "VisitMonth",
                "VisitMonthNum"
            ]:

                input_values[feature] = st.number_input(
                    feature,
                    min_value=1,
                    max_value=12,
                    value=6,
                    key=f"reg_{feature}"
                )


            elif feature == "Rating":

                input_values[feature] = st.slider(
                    "Current/Previous Rating",
                    min_value=1.0,
                    max_value=5.0,
                    value=4.0,
                    step=0.1,
                    key=f"reg_{feature}"
                )


            else:

                input_values[feature] = st.number_input(
                    feature,
                    min_value=-1,
                    value=1,
                    key=f"reg_{feature}"
                )


    if st.button(
        "Predict Rating",
        key="predict_rating"
    ):

        try:

            input_df = pd.DataFrame(
                [input_values]
            )

            # Ensure exact training feature order
            input_df = input_df[
                regression_features
            ]

            prediction = regression_model.predict(
                input_df
            )[0]

            # Rating range
            prediction = max(
                1.0,
                min(5.0, prediction)
            )

            st.success(
                f"⭐ Predicted Rating: "
                f"{prediction:.2f} / 5"
            )

        except Exception as e:

            st.error(
                "Prediction error:"
            )

            st.code(str(e))


# ============================================================
# VISIT MODE PREDICTION PAGE
# ============================================================

elif page == "🔍 Visit Mode Prediction":

    st.title("🔍 Visit Mode Prediction")

    st.write(
        "Enter tourism information to predict "
        "the visit mode."
    )

    st.divider()


    # Get exact features used during training
    if hasattr(
        classification_model,
        "feature_names_in_"
    ):

        classification_features = list(
            classification_model.feature_names_in_
        )

    else:

        classification_features = []


    st.info(
        f"Classification model expects "
        f"{len(classification_features)} features."
    )


    cls_input_values = {}


    col1, col2 = st.columns(2)


    for i, feature in enumerate(
        classification_features
    ):

        target_column = (
            col1 if i % 2 == 0 else col2
        )


        with target_column:


            if feature == "UserId":

                cls_input_values[feature] = (
                    st.number_input(
                        "User ID",
                        min_value=0,
                        value=1000,
                        key=f"cls_{feature}"
                    )
                )


            elif feature == "VisitYear":

                cls_input_values[feature] = (
                    st.number_input(
                        "Visit Year",
                        min_value=2000,
                        max_value=2030,
                        value=2025,
                        key=f"cls_{feature}"
                    )
                )


            elif feature in [
                "VisitMonth",
                "VisitMonthNum"
            ]:

                cls_input_values[feature] = (
                    st.number_input(
                        feature,
                        min_value=1,
                        max_value=12,
                        value=6,
                        key=f"cls_{feature}"
                    )
                )


            elif feature == "Rating":

                cls_input_values[feature] = (
                    st.slider(
                        "Rating",
                        min_value=1.0,
                        max_value=5.0,
                        value=4.0,
                        step=0.1,
                        key=f"cls_{feature}"
                    )
                )


            else:

                cls_input_values[feature] = (
                    st.number_input(
                        feature,
                        min_value=-1,
                        value=1,
                        key=f"cls_{feature}"
                    )
                )


    if st.button(
        "Predict Visit Mode",
        key="predict_mode"
    ):

        try:

            cls_input_df = pd.DataFrame(
                [cls_input_values]
            )

            cls_input_df = cls_input_df[
                classification_features
            ]

            prediction = (
                classification_model.predict(
                    cls_input_df
                )[0]
            )


            st.success(
                f"🔍 Predicted Visit Mode: "
                f"{prediction}"
            )

        except Exception as e:

            st.error(
                "Prediction error:"
            )

            st.code(str(e))


# ============================================================
# RECOMMENDATION PAGE
# ============================================================

elif page == "⭐ Recommendations":

    st.title("⭐ Attraction Recommendations")

    st.write(
        "Select an attraction to find similar attractions."
    )

    st.divider()


    # Check required column
    if (
        "Attraction"
        not in recommendation_df.columns
    ):

        st.error(
            "Attraction column not found in "
            "recommendation data."
        )

        st.stop()


    attraction_list = sorted(
        recommendation_df[
            "Attraction"
        ]
        .dropna()
        .unique()
    )


    selected_attraction = st.selectbox(
        "Select an Attraction",
        attraction_list
    )


    top_n = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )


    if st.button(
        "Get Recommendations"
    ):

        try:

            # Reset index to match similarity matrix
            temp_df = (
                recommendation_df
                .reset_index(drop=True)
            )


            matches = temp_df[
                temp_df["Attraction"]
                == selected_attraction
            ]


            if matches.empty:

                st.warning(
                    "Attraction not found."
                )

            else:

                idx = matches.index[0]


                similarity_scores = list(
                    enumerate(
                        similarity_matrix[idx]
                    )
                )


                # Remove selected attraction
                # and zero similarity results

                similarity_scores = [

                    (i, score)

                    for i, score in similarity_scores

                    if i != idx and score > 0

                ]


                # Sort by similarity

                similarity_scores = sorted(

                    similarity_scores,

                    key=lambda x: x[1],

                    reverse=True

                )


                top_scores = (
                    similarity_scores[:top_n]
                )


                recommendations = []


                for i, score in top_scores:

                    row = temp_df.iloc[i]


                    recommendation = {

                        "Attraction":
                        row.get(
                            "Attraction",
                            "Unknown"
                        ),

                        "Attraction Type":
                        row.get(
                            "AttractionTypeClean",
                            "Unknown"
                        ),

                        "Country":
                        row.get(
                            "Country",
                            "Unknown"
                        ),

                        "Region":
                        row.get(
                            "Region",
                            "Unknown"
                        ),

                        "Average Rating":
                        round(
                            row.get(
                                "AverageRating",
                                0
                            ),
                            2
                        ),

                        "Similarity Score":
                        round(
                            score,
                            4
                        )

                    }


                    recommendations.append(
                        recommendation
                    )


                if len(recommendations) > 0:

                    result_df = pd.DataFrame(
                        recommendations
                    )


                    st.success(
                        f"{len(result_df)} "
                        "Recommendations Found!"
                    )


                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )


                else:

                    st.warning(
                        "No similar attractions found."
                    )


        except Exception as e:

            st.error(
                "Recommendation error:"
            )

            st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌍 Tourism Experience Analytics | "
    "Machine Learning Project"
)