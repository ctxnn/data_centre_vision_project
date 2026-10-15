"""
Rock Paper Scissors vs AI.
Teach the AI your hand signs with the webcam, train it live, then play against it.

Run with:  streamlit run app.py
"""
import random

import streamlit as st
import torch
from PIL import Image

import model

EMOJI = {"rock": "✊", "paper": "✋", "scissors": "✌️"}
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}  # key beats value
COUNTER = {loser: winner for winner, loser in BEATS.items()}         # the move that beats it
MIN_CONFIDENCE = 0.6

st.set_page_config(page_title="Rock Paper Scissors vs AI", page_icon="✊")


@st.cache_resource  # load the pretrained CNN only once, not on every click
def get_cnn():
    return model.load_feature_extractor()


cnn, preprocess = get_cnn()

# st.session_state keeps values between clicks (Streamlit reruns the script on every click)
if "photos" not in st.session_state:
    st.session_state.photos = {name: [] for name in model.CLASSES}
    st.session_state.seen = set()   # ids of photos already added, so a rerun doesn't add them twice
    st.session_state.head = None    # the trained classifier layer
    st.session_state.losses = []
    st.session_state.n_train = 0
    st.session_state.score = {"You": 0, "AI": 0, "Draw": 0}
    st.session_state.last_move = None
    st.session_state.round = None


def add_photo(label, file):
    photo_id = hash(file.getvalue())
    if photo_id not in st.session_state.seen:
        st.session_state.seen.add(photo_id)
        st.session_state.photos[label].append(Image.open(file).convert("RGB"))


def play_round(image, cheat):
    probs = model.predict(cnn, preprocess, st.session_state.head, image)
    player = max(probs, key=probs.get)
    if probs[player] < MIN_CONFIDENCE:
        return {"probs": probs, "player": None}

    computer = COUNTER[player] if cheat else random.choice(model.CLASSES)
    if player == computer:
        result = "Draw"
    elif BEATS[player] == computer:
        result = "You"
    else:
        result = "AI"
    st.session_state.score[result] += 1
    return {"probs": probs, "player": player, "computer": computer, "result": result}


st.title("✊ ✋ ✌️ Rock Paper Scissors vs AI")
st.caption("Teach the AI your hand signs in one minute, then try to beat it.")

collect_tab, train_tab, play_tab = st.tabs(["1 · Collect photos", "2 · Train", "3 · Play"])

# ---------------------------------------------------------------- 1. COLLECT
with collect_tab:
    st.write("Pick a sign and take **5+ photos** of your hand showing it. "
             "Click **Clear photo** between shots. Move your hand around a little each time.")
    label = st.radio("Sign", model.CLASSES, horizontal=True,
                     format_func=lambda name: f"{EMOJI[name]} {name}")

    shot = st.camera_input("Take a photo", key="collect_cam")
    if shot is not None:
        add_photo(label, shot)

    with st.expander("No webcam? Upload photos instead"):
        for file in st.file_uploader("Photos", type=["jpg", "jpeg", "png"],
                                     accept_multiple_files=True, key=f"upload_{label}") or []:
            add_photo(label, file)

    for col, name in zip(st.columns(3), model.CLASSES):
        photos = st.session_state.photos[name]
        col.metric(f"{EMOJI[name]} {name}", f"{len(photos)} photos")
        if photos:
            col.image(photos[-1], width=140)

    if st.button("Start over"):
        st.session_state.clear()
        st.rerun()

# ---------------------------------------------------------------- 2. TRAIN
with train_tab:
    use_aug = st.checkbox("Use data augmentation (mirror, rotate, darker, brighter)", value=True)
    epochs = st.slider("Epochs", 10, 300, 100)
    lr = st.select_slider("Learning rate", [0.0001, 0.001, 0.01, 0.1, 1.0], value=0.01)

    if st.button("Train the AI", type="primary"):
        counts = [len(st.session_state.photos[name]) for name in model.CLASSES]
        if min(counts) < 2:
            st.warning("Take at least 2 photos of each sign first.")
        else:
            images, labels = [], []
            for class_index, name in enumerate(model.CLASSES):
                for photo in st.session_state.photos[name]:
                    versions = model.augment(photo) if use_aug else [photo]
                    images += versions
                    labels += [class_index] * len(versions)

            with st.spinner("Extracting features with MobileNet and training..."):
                features = model.get_features(cnn, preprocess, images)
                head, losses = model.train_classifier(features, torch.tensor(labels), epochs, lr)

            st.session_state.head = head
            st.session_state.losses = losses
            st.session_state.n_train = len(images)
            st.session_state.score = {"You": 0, "AI": 0, "Draw": 0}

    if st.session_state.head is not None:
        st.success(f"Trained on {st.session_state.n_train} images "
                   f"({sum(len(p) for p in st.session_state.photos.values())} photos you took). "
                   "Go to **3 · Play**!")
        st.write("**Training loss** (lower = fewer mistakes on the training photos)")
        st.line_chart(st.session_state.losses, x_label="epoch", y_label="loss")

# ---------------------------------------------------------------- 3. PLAY
with play_tab:
    if st.session_state.head is None:
        st.info("Train the AI first (step 2).")
    else:
        cheat = st.toggle("😈 Cheat mode: the AI looks at your hand before choosing")
        move = st.camera_input("Show your move!", key="play_cam")

        # only play a new round for a new photo (Streamlit reruns on every click)
        if move is not None and hash(move.getvalue()) != st.session_state.last_move:
            st.session_state.last_move = hash(move.getvalue())
            st.session_state.round = play_round(Image.open(move).convert("RGB"), cheat)

        rnd = st.session_state.round
        if rnd is not None:
            st.write("**What the AI sees:**")
            for name in model.CLASSES:
                st.progress(rnd["probs"][name], text=f"{EMOJI[name]} {name}: {rnd['probs'][name]:.0%}")

            if rnd["player"] is None:
                st.warning(f"The AI isn't sure (below {MIN_CONFIDENCE:.0%} confident). Try again!")
            else:
                st.subheader(f"You {EMOJI[rnd['player']]}  vs  {EMOJI[rnd['computer']]} AI")
                if rnd["result"] == "You":
                    st.success("You win! 🎉")
                elif rnd["result"] == "AI":
                    st.error("AI wins! 🤖")
                else:
                    st.info("Draw!")

        for col, (who, points) in zip(st.columns(3), st.session_state.score.items()):
            col.metric(who, points)
