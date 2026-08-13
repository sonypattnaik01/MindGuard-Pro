# generate_multimodal_fusion_diagram.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Text Branch
rect_text = FancyBboxPatch((0.5, 7), 3, 1.5, boxstyle="round,pad=0.05",
                            facecolor='#667eea', edgecolor='white', linewidth=2)
ax.add_patch(rect_text)
ax.text(2, 8.2, 'TEXT BRANCH', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
ax.text(2, 7.6, 'T5 Encoder → [CLS]', ha='center', va='center', fontsize=8, color='white')

# Face Branch
rect_face = FancyBboxPatch((5, 7), 3, 1.5, boxstyle="round,pad=0.05",
                            facecolor='#48bb78', edgecolor='white', linewidth=2)
ax.add_patch(rect_face)
ax.text(6.5, 8.2, 'FACE BRANCH', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
ax.text(6.5, 7.6, 'ViT → 7 Emotion Probs', ha='center', va='center', fontsize=8, color='white')

# Behavioral Branch
rect_behav = FancyBboxPatch((9.5, 7), 3, 1.5, boxstyle="round,pad=0.05",
                             facecolor='#ecc94b', edgecolor='white', linewidth=2)
ax.add_patch(rect_behav)
ax.text(11, 8.2, 'BEHAVIORAL BRANCH', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
ax.text(11, 7.6, 'Sleep + Mood + Notes', ha='center', va='center', fontsize=8, color='white')

# Arrows to fusion
for x in [2, 6.5, 11]:
    ax.annotate('', xy=(x, 6.5), xytext=(x, 7), arrowprops=dict(arrowstyle='->', color='gray', lw=2))

# Fusion Module
rect_fusion = FancyBboxPatch((2.5, 4), 8, 2, boxstyle="round,pad=0.05",
                              facecolor='#f56565', edgecolor='white', linewidth=2)
ax.add_patch(rect_fusion)
ax.text(6.5, 5.6, 'PROMPT-BASED MULTIMODAL FUSION', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
ax.text(6.5, 5, '"Analyze depression risk based on:\nText: {text}\nFacial Emotions: {summary}\nSleep: {hours}\nMood: {score}"', 
        ha='center', va='center', fontsize=7, color='white')

# Arrow to decoder
ax.annotate('', xy=(6.5, 3.5), xytext=(6.5, 4), arrowprops=dict(arrowstyle='->', color='gray', lw=2))

# T5 Decoder
rect_decoder = FancyBboxPatch((3, 1.5), 7, 1.5, boxstyle="round,pad=0.05",
                               facecolor='#9f7aea', edgecolor='white', linewidth=2)
ax.add_patch(rect_decoder)
ax.text(6.5, 2.8, 'T5 DECODER + CLASSIFICATION HEAD', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
ax.text(6.5, 2.2, '12 Transformer Layers → Linear(768→256) → Linear(256→3)', ha='center', va='center', fontsize=7, color='white')

# Arrow to output
ax.annotate('', xy=(6.5, 0.8), xytext=(6.5, 1.5), arrowprops=dict(arrowstyle='->', color='gray', lw=2))

# Output
rect_output = FancyBboxPatch((3.5, 0), 6, 0.7, boxstyle="round,pad=0.05",
                              facecolor='#805ad5', edgecolor='white', linewidth=2)
ax.add_patch(rect_output)
ax.text(6.5, 0.35, 'OUTPUT: Low Risk (12%) | Moderate Risk (68%) | High Risk (20%)', 
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Title
ax.text(6.5, 9.5, 'MindGuard Pro: Multimodal Fusion Architecture', 
        ha='center', va='center', fontsize=16, fontweight='bold', color='#1a202c')

plt.tight_layout()
plt.savefig('multimodal_fusion_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print('✅ multimodal_fusion_diagram.png generated')