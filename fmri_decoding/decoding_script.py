import pandas as pd
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from nilearn.image import index_img
from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# =====================================================================
# 1. LOAD THE DATASET
# =====================================================================
print("Loading Haxby dataset...")
haxby_dataset = datasets.fetch_haxby()
func_filename = haxby_dataset.func[0]
mask_filename = haxby_dataset.mask_vt[0]

# Load behavioral logs (labels and scanner run chunks)
behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=" ")
labels = behavioral["labels"]
runs = behavioral["chunks"]

# =====================================================================
# 2. FILTER OUT THE 'REST' CONDITION
# =====================================================================
# Create a boolean mask where True means an active visual task block
condition_mask = labels != "rest"

# Apply the mask uniformly across all components so shapes match perfectly
fmri_niimgs = index_img(func_filename, condition_mask)  # Filtered 4D brain scan
y_true_labels = labels[condition_mask]                  # Filtered target labels
run_groups = runs[condition_mask]                       # Filtered cross-val runs

# =====================================================================
# 3. INITIALIZE AND TRAIN THE DECODER
# =====================================================================
print("Initializing Decoder and training SVM model...")
decoder = Decoder(
    estimator="svc",
    mask=mask_filename,
    standardize="zscore_sample",
    cv=LeaveOneGroupOut(),
    screening_percentile=20
)

# Fit the decoder using the filtered data
decoder.fit(fmri_niimgs, y_true_labels, groups=run_groups)

# =====================================================================
# 4. GENERATE PREDICTIONS AND CHECK ACCURACY
# =====================================================================
# Use the trained model to predict the visual category from the brain images
y_pred = decoder.predict(fmri_niimgs)

# Calculate the final overall accuracy percentage
final_accuracy = accuracy_score(y_true_labels, y_pred)

print("\n=========================================")
print(f"🎉 Final Pipeline Accuracy: {final_accuracy * 100:.2f}%")
print("=========================================")

# Print internal cross-validation accuracies per category
print("\nMean Cross-Validation Accuracy per category:")
for category in decoder.classes_:
    mean_score = decoder.cv_scores_[category].mean()
    print(f"  {category}: {mean_score * 100:.2f}%")

# =====================================================================
# 5. VISUALIZATION DELIVERABLES
# =====================================================================
print("\nGenerating visual plots...")

# Deliverable A: Plot the Confusion Matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_true_labels,
    y_pred,
    display_labels=decoder.classes_,
    cmap=plt.cm.Blues,
    xticks_rotation=45
)
plt.title("Decoder Confusion Matrix")
plt.tight_layout()

# Deliverable B: Plot the Brain Mapping Weight Clusters
weight_img = decoder.coef_img_["face"]
plotting.plot_stat_map(
    weight_img,
    bg_img=haxby_dataset.anat[0],
    title="SVM Weights for 'Face' Discrimination",
    display_mode="z",
    cut_coords=[-15, -12, -9],
    colorbar=True
)

# Render both open plots to your screen
plt.show()
