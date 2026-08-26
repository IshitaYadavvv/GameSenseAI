import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(
    page_title="GameSense AI",
    page_icon="🎮",
    layout="wide"
)

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model("gamesense_dnn.keras")
    te = joblib.load("target_encoder.pkl")
    ohe = joblib.load("onehot_encoder.pkl")
    scaler = joblib.load("numerical_scaler.pkl")
    dnn_scaler = joblib.load("dnn_scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, te, ohe, scaler, dnn_scaler, feature_columns

@st.cache_data
def load_data():
    return pd.read_csv("gamesense_clean_dataset.csv")

model, te, ohe, scaler, dnn_scaler, feature_columns = load_models()
df = load_data()

gpu_years = df.groupby("GPU")["GPU_Release_Year"].first().to_dict()
game_years = df.groupby("Game")["Game_Release_Year"].first().to_dict()

gpus = sorted(df["GPU"].dropna().unique())
games = sorted(df["Game"].dropna().unique())
resolutions = list(df["Resolution"].dropna().unique())
qualities = list(df["Graphics_Quality"].dropna().unique())

nvidia_gpus = sorted(
    [gpu for gpu in gpus if "NVIDIA" in gpu.upper()]
)

resolution_priority = {
    "1920x1080": 1,
    "2560x1440": 2,
    "3440x1440": 3,
    "3840x2160": 4
}

quality_priority = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Ultra": 4
}

def predict_fps(gpu, game, resolution, graphics_quality):

    user_data = pd.DataFrame({
        "GPU": [gpu],
        "Game": [game],
        "Resolution": [resolution],
        "Graphics_Quality": [graphics_quality],
        "GPU_Release_Year": [gpu_years[gpu]],
        "Game_Release_Year": [game_years[game]]
    })

    user_te = te.transform(
        user_data[["GPU", "Game"]]
    )

    user_ohe = pd.DataFrame(
        ohe.transform(
            user_data[["Resolution", "Graphics_Quality"]]
        ),
        columns=ohe.get_feature_names_out(
            ["Resolution", "Graphics_Quality"]
        ),
        index=user_data.index
    )

    num_cols = [
        "GPU_Release_Year",
        "Game_Release_Year"
    ]

    user_num = pd.DataFrame(
        scaler.transform(user_data[num_cols]),
        columns=num_cols,
        index=user_data.index
    )

    user_final = pd.concat(
        [user_te, user_ohe, user_num],
        axis=1
    )

    user_final = user_final[feature_columns]

    user_dnn = dnn_scaler.transform(
        user_final.values
    )

    predicted_log_fps = model.predict(
        user_dnn,
        verbose=0
    ).flatten()[0]

    return max(
        0,
        float(np.expm1(predicted_log_fps))
    )

def performance_rating(fps):

    if fps < 30:
        return "🔴 Poor"

    elif fps < 60:
        return "🟠 Playable"

    elif fps < 100:
        return "🟢 Good"

    elif fps < 144:
        return "🔵 Excellent"

    else:
        return "🟣 Very High"

def optimize_for_target_fps(gpu, game, target_fps):

    results = []

    for resolution in resolutions:

        for quality in qualities:

            try:

                fps = predict_fps(
                    gpu,
                    game,
                    resolution,
                    quality
                )

                results.append({
                    "Resolution": resolution,
                    "Graphics Quality": quality,
                    "Predicted FPS": fps
                })

            except Exception:
                pass

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return None, results_df

    results_df["Resolution Score"] = (
        results_df["Resolution"]
        .map(resolution_priority)
        .fillna(0)
    )

    results_df["Quality Score"] = (
        results_df["Graphics Quality"]
        .map(quality_priority)
        .fillna(0)
    )

    suitable = results_df[
        results_df["Predicted FPS"] >= target_fps
    ].copy()

    if suitable.empty:
        return None, results_df

    suitable = suitable.sort_values(
        by=[
            "Resolution Score",
            "Quality Score",
            "Predicted FPS"
        ],
        ascending=[
            False,
            False,
            False
        ]
    ).reset_index(drop=True)

    return suitable.iloc[0], suitable

def compare_nvidia_gpus(
    game,
    resolution,
    graphics_quality
):

    results = []

    for gpu in nvidia_gpus:

        try:

            fps = predict_fps(
                gpu,
                game,
                resolution,
                graphics_quality
            )

            results.append({
                "GPU": gpu,
                "Predicted FPS": fps
            })

        except Exception:
            pass

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df

    return results_df.sort_values(
        by="Predicted FPS",
        ascending=False
    ).reset_index(drop=True)

st.markdown(
    """
    <h1 style="text-align:center;">
        🎮 GameSense AI
    </h1>

    <p style="text-align:center; font-size:20px;">
        Deep Learning-Based Gaming Performance Prediction
        & Optimization
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

tab1, tab2, tab3 = st.tabs([
    "🎯 Predict FPS",
    "🧠 Smart Optimizer",
    "🏆 NVIDIA Comparison"
])

with tab1:

    st.subheader("🎯 Predict Average FPS")

    col1, col2 = st.columns(2)

    with col1:

        selected_gpu = st.selectbox(
            "Select GPU",
            gpus
        )

        selected_game = st.selectbox(
            "Select Game",
            games
        )

    with col2:

        selected_resolution = st.selectbox(
            "Select Resolution",
            resolutions
        )

        selected_quality = st.selectbox(
            "Select Graphics Quality",
            qualities
        )

    if st.button(
        "🎮 Predict FPS",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing gaming configuration..."
        ):

            fps = predict_fps(
                selected_gpu,
                selected_game,
                selected_resolution,
                selected_quality
            )

        rating = performance_rating(fps)

        st.success("Prediction completed!")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Average FPS",
                f"{fps:.2f} FPS"
            )

        with col2:

            st.metric(
                "Performance",
                rating
            )

with tab2:

    st.subheader(
        "🧠 Smart Target-FPS Optimizer"
    )

    st.write(
        "Find the highest-quality configuration "
        "predicted to achieve your target FPS."
    )

    col1, col2 = st.columns(2)

    with col1:

        optimizer_gpu = st.selectbox(
            "Select GPU",
            gpus,
            key="optimizer_gpu"
        )

        optimizer_game = st.selectbox(
            "Select Game",
            games,
            key="optimizer_game"
        )

    with col2:

        target_fps = st.slider(
            "Target FPS",
            30,
            240,
            60,
            10
        )

    if st.button(
        "🧠 Find Best Configuration",
        use_container_width=True
    ):

        with st.spinner(
            "Testing gaming configurations..."
        ):

            best_config, suitable_configs = (
                optimize_for_target_fps(
                    optimizer_gpu,
                    optimizer_game,
                    target_fps
                )
            )

        if best_config is not None:

            st.success(
                "Recommended configuration found!"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Resolution",
                    best_config["Resolution"]
                )

            with col2:

                st.metric(
                    "Graphics Quality",
                    best_config[
                        "Graphics Quality"
                    ]
                )

            with col3:

                st.metric(
                    "Predicted FPS",
                    f"{best_config['Predicted FPS']:.2f}"
                )

            st.subheader(
                "Configurations Meeting Target FPS"
            )

            display_df = suitable_configs[
                [
                    "Resolution",
                    "Graphics Quality",
                    "Predicted FPS"
                ]
            ].copy()

            display_df["Predicted FPS"] = (
                display_df["Predicted FPS"].round(2)
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No available configuration is "
                "predicted to reach your target FPS."
            )

with tab3:

    st.subheader(
        "🏆 NVIDIA GPU Comparison"
    )

    col1, col2 = st.columns(2)

    with col1:

        comparison_game = st.selectbox(
            "Select Game",
            games,
            key="comparison_game"
        )

    with col2:

        comparison_resolution = st.selectbox(
            "Select Resolution",
            resolutions,
            key="comparison_resolution"
        )

    comparison_quality = st.selectbox(
        "Select Graphics Quality",
        qualities,
        key="comparison_quality"
    )

    if st.button(
        "🏆 Compare NVIDIA GPUs",
        use_container_width=True
    ):

        with st.spinner(
            "Comparing NVIDIA GPUs..."
        ):

            comparison_results = (
                compare_nvidia_gpus(
                    comparison_game,
                    comparison_resolution,
                    comparison_quality
                )
            )

        if not comparison_results.empty:

            winner = comparison_results.iloc[0]

            st.success(
                f"🏆 Best predicted performance: "
                f"{winner['GPU']} — "
                f"{winner['Predicted FPS']:.2f} FPS"
            )

            display_df = comparison_results.copy()

            display_df["Predicted FPS"] = (
                display_df["Predicted FPS"].round(2)
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                comparison_results.head(10)
                .set_index("GPU")["Predicted FPS"]
            )

        else:

            st.warning(
                "No NVIDIA GPU predictions "
                "could be generated."
            )

st.divider()

st.caption(
    "GameSense AI uses regression and deep learning "
    "to predict gaming performance and support "
    "configuration decisions."
)