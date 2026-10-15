# ✊ ✋ ✌️ Rock Paper Scissors vs AI

Teach an AI your hand signs with a webcam in about one minute, train it live in front of the class, then play against it.

**Course concept shown:** transfer learning (Module 2), plus network building blocks and tensor shapes (Module 3), data augmentation (Module 4), and loss, optimizer and softmax confidence (Module 5).

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The browser opens at http://localhost:8501. Allow camera access when asked.
The first launch downloads MobileNet (~10 MB), so **run it once before the demo** while you have internet.

## Files

| File | What it does |
|---|---|
| `model.py` | All the AI: pretrained CNN, augmentation, training, prediction (~70 lines) |
| `app.py` | The Streamlit screens and the game rules |

## How it works

```
webcam photo ──► MobileNetV3 (pretrained, frozen) ──► 576 numbers ──► Linear layer (we train this) ──► softmax ──► rock / paper / scissors
                  "knows how to see"                  feature vector      576 → 3                         probabilities
```

1. **Collect.** Take ~5 photos for each sign.
2. **Augment.** Each photo becomes 5: original, mirrored, rotated 15°, darker, brighter. So 15 photos turn into 75 training images.
3. **Extract features.** MobileNetV3, already trained on 1.2 million ImageNet photos, turns each image into 576 numbers. We removed its last layer (`classifier = nn.Identity()`), so it describes the image instead of naming one of its 1000 ImageNet classes.
4. **Train.** Only one new `Linear(576, 3)` layer is trained (1,731 parameters) using CrossEntropy loss and the Adam optimizer. It takes about a second.
5. **Play.** The AI reads your hand. If it is less than 60% sure, it asks you to try again. Otherwise the computer picks a random move and the scoreboard updates.
   **Cheat mode** makes the computer pick the move that beats yours, which shows the AI really is reading your hand.

## Presentation split (4 people, ~1–2 min each)

**Person 1: The problem and the idea (Module 1)**
- Task: classify a hand image into 3 classes. That makes it **image classification**, not detection or segmentation.
- Challenge: we only have a few photos. Training a CNN from scratch would need thousands.
- Solution: **transfer learning**. Reuse a network that already knows how to see.

**Person 2: The model (Modules 2–3), `load_feature_extractor` and `train_classifier` in `model.py`**
- MobileNetV3-Small is a lightweight CNN designed for phones and edge devices (fits the course name).
- Tensor shapes: image `(3, 224, 224)` → CNN → `(576,)` → Linear → `(3,)`.
- We **freeze** the CNN and train only the last layer: 1,731 parameters, while the frozen CNN has ~927,000.

**Person 3: Data augmentation (Module 4), `augment` in `model.py`**
- Mirror (left vs right hand), rotate (tilted hand), brightness (different lighting).
- 5x more training data from the same photos.
- Demo tip: train once **without** augmentation, then switch the lights or use the other hand. It makes more mistakes. Retrain **with** augmentation and it does better.

**Person 4: Training and confidence (Module 5), the Train and Play tabs**
- Loss curve: CrossEntropy loss drops toward 0 as the layer learns.
- Optimizer: Adam. Learning rate slider: try 1.0 (loss jumps around) vs 0.01 (smooth).
- Softmax turns scores into probabilities. The bars in the Play tab show them.
- Below 60% confidence, the AI says "not sure" instead of guessing.

## Live demo script (~3 min)

1. Tab 1: take 5 photos each of rock, paper and scissors (move your hand a bit between shots).
2. Tab 2: keep augmentation on, click **Train**, and show the loss curve.
3. Tab 3: play 3–4 rounds and point at the confidence bars.
4. Turn on **Cheat mode**: the AI wins every round. That proves it really recognizes your hand.
5. Optional: show the learning rate or augmentation experiment from above.

## Likely viva questions

- **Why not train the whole CNN?** Too few photos. It would overfit, and it would be slow on a laptop.
- **Why MobileNet?** It is small and fast, built for edge devices like phones and a Jetson.
- **What is the feature vector?** The CNN's summary of the image (shapes, edges, textures) as 576 numbers.
- **Why softmax?** It turns raw scores into probabilities that add up to 1, so we can measure confidence.
- **Why does the loss reach ~0 but the AI still makes mistakes?** The loss is measured on training photos. New photos (different lighting or angle) can still fool it. That's why augmentation and more varied photos help.
- **What would you improve?** More photos from different people and backgrounds, a separate validation set, and fine-tuning the last few CNN layers.

## Tips for demo day

- Use a plain background (a wall or a table) and keep your hand fairly large in the frame.
- Take the training and playing photos in the same spot.
- No webcam working? Use the "Upload photos instead" option in Tab 1.
