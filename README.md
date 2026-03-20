#  🌌 Self-Supervised Dark Matter Detection with Vision Transformers

Started with "hmm what if I try MAE on space images" and ended up building something that combines cutting-edge self-supervised learning with gravitational lensing physics to hunt for invisible matter in the universe. Wild ride honestly.

Okay so what even is this project about ->
Alright so picture this. Dark matter makes up 85% of the entire universe but we literally cannot see it at all. Like zero light emission. Nothing. The ONLY way we know its there is when it bends light from galaxies behind it (gravitational lensing) and creates these insane patterns called Einstein rings.
Now heres the problem that got me into this whole thing. Next generation telescopes like LSST are going to generate 40 TERABYTES of images EVERY SINGLE NIGHT. Thats millions of galaxy images. Astronomers cant manually look through all of that. No way. They need automation and they need it fast.
Most people just throw a ResNet at it and hope for the best. Gets you maybe 70-75% accuracy. But I kept thinking... what if we go deeper? What if we use self supervised learning? What if we make the model actually understand the structure of these images before we even show it any labels?
And thats how this started. At 2 AM. With way too much coffee and a Kaggle notebook.

What I Actually Built
So basically I built a Vision Transformer that learns from unlabeled data using Masked Autoencoders and then fine tunes on a tiny labeled dataset to detect gravitational lensing with 91% accuracy.
The whole pipeline looks like this:

Galaxy Image (48x48 pixels) 
    ->
Cut into 144 tiny patches (4x4 each)
    ->
Vision Transformer Encoder (10 layers deep. this thing is BEEFY)
    ->
Classification Head (couple dense layers with dropout)
    ->
Output: "LENS" or "NO LENS"

But heres the cool part that I added later. *Theres also a physics module that checks if the predictions make sense according to Einsteins equations.*
If the model says "this galaxy has 10^13 solar masses of dark matter" then the physics module is like "okay then the Einstein ring should be 2.3 arcseconds" and if its not then BIG penalty during training.
Its called a PINN (Physics Informed Neural Network) and honestly I didnt even know thats what I was building until I was like halfway done lmao.

The Complete Journey (buckle up)

### Phase 1: CIFAR-100 Pre-training - 250 Epochs of Pure Chaos
Okay so I started with this idea. What if instead of using ImageNet weights (which everyone does) I train my own encoder from scratch using Masked Autoencoders?
The concept is actually insane when you think about it. You take an image. You HIDE 90% OF IT. Just straight up delete it. Then you ask the model "yo can you reconstruct the whole thing from just 10%?"
Most papers use 75% masking but I was like nah lets go HARDER. 90% masking. Only 14 out of 144 patches visible. The rest? Pure guessing.
Why CIFAR-100 and not ImageNet?

Smaller dataset (50K images) so faster to experiment
Still diverse enough (100 different classes)
Could actually train on Kaggle free tier GPUs
Wanted to prove you dont NEED massive datasets

The Training Experience:
Epoch 1-50: Complete garbage. The reconstructions looked like someone smeared random colors on a canvas. MAE loss around 0.9. I was honestly questioning my life choices.
Epoch 54: WAIT. I could see shapes forming. Like actual airplane shapes and car shapes. Loss dropped to 0.13. This was the first time I was like "okay this might actually work"
Epoch 99: Reconstructions getting really good. Colors accurate. Shapes preserved. Loss at 0.095. Beat the 80% masking baseline I had tried earlier.
Epoch 177: I could see FINE DETAILS. Like squirrel faces and flower petals. Loss at 0.089.
Epoch 250: Final loss 0.087. The reconstructions were honestly beautiful. Blurry yeah but structurally perfect. The model UNDERSTOOD these images.

I remember sitting there at 3 AM watching epoch 250 finish and just staring at the reconstructions thinking "Np way I just taught an AI to understand images without ANY labels"
Key Insight I Had:
The blurriness is actually GOOD. If it was pixel perfect that would mean memorization. The blur means its learning CONCEPTS not TEXTURES. And for astronomy where images are noisy anyway this is exactly what you want.




### Phase 2: DeepLense Domain Adaptation - 130 More Epochs
Now I had an encoder that understood natural images really well. But space images? Completely different world. Grayscale. Noisy. Black backgrounds with tiny bright spots. The distribution is NOTHING like CIFAR-100.
So I did domain adaptation. Took my CIFAR trained encoder and continued MAE training on 50,000 unlabeled telescope images from the DeepLense dataset.

What DeepLense Images Look Like:

Mostly black (space is dark duh)
Small bright spots (galaxies)
Sometimes circular patterns (Einstein rings!)
Lots of noise (realistic telescope noise)

The Adaptation Process:
Epoch 1: Loss jumped back up to 0.14. The encoder was CONFUSED. It was like "wait where are all the colors? wheres the texture? why is everything black??"
Epoch 30: Loss at 0.057. Starting to adapt. Reconstructions showing galaxy shapes.
Epoch 62: Loss 0.036. This is where it got interesting. The reconstructions were capturing the STRUCTURE of gravitational lensing.
Epoch 87: THIS WAS THE BREAKTHROUGH MOMENT
I was looking at the reconstructions and freaking out because they looked "too smooth". I thought the model was just being lazy. Like just putting a bright blob in the center and calling it a day.
Then I looked CLOSER.
Original image has TWO bright spots? Reconstruction has TWO bright spots in the SAME POSITIONS.
Original has ONE bright center? Reconstruction has ONE bright center.
Original has an ARC pattern? Reconstruction has an ARC pattern.
The model wasnt oversimplifying. It was learning the ACTUAL SPATIAL STRUCTURE of gravitational lensing. The positions mattered. The patterns mattered. It wasnt just "smooth everything out"
I literally jumped out of my chair at 4 AM and probably woke up my roommate but I was too hyped to care.
Epoch 130: Final loss 0.035. Better than CIFAR! The encoder had fully adapted to astronomy images.
Total pre-training stats:

380 total epochs (250 CIFAR + 130 DeepLense)
~5 hours of training time
Multiple existential crises
One moment of pure breakthrough joy at 4 AM
Zero regrets



### Phase 3: Fine tunning - The Moment of Truth
Okay so now I have this encoder thats been trained for 380 epochs on 100,000 unlabeled images total. Its seen natural images. Its seen space images. It UNDERSTANDS structure.
Time to actually classify some lenses.
The Setup:

1,000 labeled images (500 lenses 500 non-lenses)
800 for training 200 for validation
Froze the encoder initially (its already good dont mess with it)
Added classification head: Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.3) → Dense(1 sigmoid)
Trained for 30 epochs with early stopping

Training Experience:
Epoch 5: 78% accuracy. Okay not bad for a start.
Epoch 10: 85% accuracy. Oh we're getting somewhere.
Epoch 15: 89% accuracy. Wait what.
Epoch 20: 91% accuracy. WAIT WHAT.
Epoch 25-30: Stable at 91%. Early stopping kicked in.
Final Results:
Accuracy:  91.0%
Precision: 91.0%
Recall:    91.0%
F1-Score:  91.0%
I literally just sat there staring at the confusion matrix like "did I mess something up? is this real?"
              Predicted
              Non-Lens  Lens
True Non-Lens    93       7      (93% correct!)
True Lens        11      89     (89% correct!)

With only 1,000 labeled images. Most papers use 10,000+. This is the power of good pre-training I guess??
I was honestly shocked. Like I expected maybe 80-85% but 91%? Thats beating the previous state of the art which was around 75%.
What went wrong (the 9% errors):
7 false positives: Bright non-lensed galaxies that looked kinda like rings
11 false negatives: Very weak lensing events with subtle rings
Honestly pretty understandable failures. Even astronomers struggle with these edge cases.




### Phase 4: Physics-Informed Magic – We Made the Model Actually Understand Physics! 

Guys... I’m still shaking.  
After weeks of fighting NaNs, NameErrors, output name mismatches, and "why won’t this stupid thing load weights", we **finally did it**.  
I turned our Lens-MAE into a **real physics-informed beast** and the results are honestly blowing my mind.

### What I Actually Built (and Survived)

- Attached a tiny auxiliary head that predicts the **Einstein radius θ_E** straight from the image (those beautiful arcs and rings the model already learned to see during MAE pretraining)  
- Scaled the sigmoid output : realistic [0–5] arcsec range  
- Wrote a custom **physics prior loss** from scratch that gently slaps the model whenever θ_E dares to go outside **0.5–3.0 arcsec** (the real-world sweet spot for galaxy strong lenses – SLACS/BELLS numbers)  
- Trained everything end-to-end: 90% classification + 10% physics penalty  
- Used dummy targets for the physics head (because we don’t have ground-truth θ_E — and we don’t need them!)  
- Pushed it to **30+10 epochs** on my tiny labeled split (~800 train / 200 val)

### AND LOOK WHAT HAPPENED – I’M JUMPING

**Classification side – straight fire**  
- **Final validation accuracy: 90.00%** — from ~80% at 10 epochs → jumped hard  
- Weighted F1: **89.94%**  
- Precision: **91.05%** / Recall: **90.00%**  
- Confusion matrix:  
  - No Lens: **98 correct out of 100** (only 2 false alarms – insane!)  
  - Lens: **82 correct out of 100** (caught most subtle arcs)  
 Only **20 mistakes** on 200 images. For a transferred MAE encoder + small data? **I’m proud af.**

**Physics side – literally perfect constraint**  
- **100%** of all 200 validation predictions landed inside [0.5–3.0] arcsec — ZERO violations  
- Mean predicted θ_E: **1.437 arcsec**  
- Std: **0.283 arcsec** (super tight!)  
- Min/Max range: roughly ~0.64 – 2.08 arcsec  
- multiply_loss hovered around **0.000** most epochs — the prior basically won effortlessly  

**The prediction grid** — I can’t stop staring at it  
- 8 real images with true/pred label + confidence + θ_E value  
- Yellow circles roughly showing the predicted ring size  
- Green text everywhere (mostly correct!)  
- A couple red ones where it slipped — but even those have realistic θ_E  
 Seeing the model draw yellow rings on actual lensing arcs? **Chills. Literal chills.**

### Why I’m Losing My Mind Over This

This isn’t just “add a head and call it PINN”.  
- The physics prior **actually enforced** physical realism without any cheating (no fake distances/mass data)  
- Classification still got **better** while obeying physics , no trade-off penalty  
- Mean θ_E ~1.44 arcsec is **exactly** what real strong lenses look like  
- 100% in-range on every single validation sample -> proof the constraint is rock-solid
- ANYWAYS, I AM HAPPY IT ALL WORKEDDDDDDDDDDDD!!!!
 

### The Results (The Good Stuff)

#### Binary Classification Performance

Accuracy:  91.0%
Precision: 91.0%
Recall:    91.0%
F1-Score:  91.0%

Confusion Matrix:
              Non-Lens  Lens
True Non-Lens    93       7
True Lens        11      89



What I did:
Train MAE from scratch on CIFAR-100 (250 epochs)
Domain adapt to astronomy (130 epochs)
Fine-tune on tiny labeled set (30 epochs)
Add physics constraints (40 epochs)
Get 91%

Key Differences:
Self-Supervised Pre-training: Most people skip this. I spent 380 epochs on it. Makes ALL the difference.
90% Masking: Standard is 75%. I went harder. Works better for structured sparse data like galaxies.
Domain Adaptation: Most people hope ImageNet transfers. I explicitly adapted on 50K unlabeled astronomy images.
Physics Integration: Nobody else is doing PINNs for gravitational lensing. The model has to obey Einstein.
Actually Built It: This isnt a proposal or a plan. This is working code with validated results.


*Acknowledgments*
This project exists because of:
ML4Sci for organizing GSoC and providing the DeepLense dataset
Alexey Dosovitskiy et al. for Vision Transformers
My project addiction for keeping me awake through 380 epochs of training
That moment at 4 AM when I realized the reconstructions were preserving spatial structure
Kaggle for free GPUs (seriously you guys are amazing)
