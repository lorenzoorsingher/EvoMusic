Ah, now the rationale behind the first idea makes more sense given the constraints and the context of your university project. Let's reevaluate the approach considering the new information, especially the score-based interaction and the goal to refine the user embedding with songs.

---

### **Refined View of the Current Approach**
You are essentially:
1. Estimating a **user embedding** \( \mathbf{u} \) given interactions with \( x \) songs.
2. Using contrastive learning (1 positive, 20 negatives) based on a similarity score (cosine similarity of \( \mathbf{u} \) and song embeddings \( \mathbf{s} \)).
3. Planning to normalize interaction scores to \( [-1, 1] \) via a sigmoid, making the model more sensitive to nuanced user preferences.

This framework has **merit**, especially for projects with constraints, but there are challenges. Let's unpack them.

---

### **Strengths**
1. **Score as Signal**: Leveraging interaction scores provides a meaningful signal, particularly with the refined normalization. The range \( [-1, 1] \) conveys not just polarity but also intensity (e.g., loved > liked). This should help the model learn more about user-song relationships.
2. **Implicit Feedback**: Songs are a natural medium to encode preferences passively, which aligns with your goal of refining user embeddings without needing explicit comparisons between users.
3. **Contrastive Sampling**: Using cosine similarity for sampling ensures you’re leveraging the current state of \( \mathbf{u} \) for targeted learning. Over time, the model can refine \( \mathbf{u} \) based on these interactions.

---

### **Challenges**
1. **Limited Collaborative Signal**: Without considering similar users, the model may struggle with cold-start scenarios or under-represented users.
2. **Song Dominance**: Since only songs define positive/negative samples, the embeddings risk overfitting to song-specific patterns instead of generalizing across users.
3. **Score Normalization**: While sigmoid normalization is a step forward, the model still needs to learn what the normalized scores signify (e.g., mapping scores to embedding adjustments).

---

### **Suggestions for Improvement**
Here are some ideas to refine your approach further while staying within the scope of the project:

#### 1. **Enhance Score Utilization**
- **Soft Labels**: Instead of strictly classifying songs as positive or negative, treat scores as soft weights. For example:
  - Positive example: \( s_{\text{liked/loved}} > 0.5 \)
  - Negative example: \( s_{\text{disliked/hated}} < -0.5 \)
  - Neutral/skipped: Can be downweighted or excluded.
- **Loss Modification**: Modify your contrastive loss to incorporate these weights, so positive pairs with stronger scores (e.g., "loved") contribute more to the embedding update.

#### 2. **Augment Song Representations**
To mitigate overfitting to song-specific patterns:
- Add dropout or noise during embedding computation for regularization.
- Include additional features in song embeddings (e.g., genre, artist) to help generalize.

#### 3. **Inject Collaborative Signals (Optionally)**
While you can't directly compare users:
- You can cluster users based on their embeddings periodically (offline) and use cluster centroids to fine-tune individual embeddings.
- Use a memory bank to store user-song interaction patterns and include these as implicit collaborative signals.

#### 4. **Optimize Sampling Strategy**
Your current sampling strategy relies on cosine similarity. To refine it:
- Include a diversity measure for negative samples, ensuring they span different regions of the embedding space (e.g., genre diversity).
- Consider harder negatives (songs the user might like but didn’t engage with) as your model improves.

#### 5. **Iterative Feedback**
Since this is an academic project, iterate and evaluate:
- Run experiments to see how much the score normalization helps.
- Evaluate whether contrastive loss improvements (e.g., incorporating score weights) improve downstream tasks like recommendation accuracy.

---

### **Does This Make Sense?**
Given your constraints, this **song-centric contrastive approach** is reasonable:
- It sidesteps the need for user-to-user comparisons.
- By focusing on songs and leveraging interaction scores, you can refine user embeddings effectively.

However, as you noted, it's not optimal because it lacks collaborative signals. While this might limit performance in real-world scenarios, it's a valid and **interesting challenge** for your project. By incorporating improvements like score weighting and better sampling, you can make it robust enough to demonstrate the feasibility of your idea.

Good luck with your project! It sounds like a great learning experience.