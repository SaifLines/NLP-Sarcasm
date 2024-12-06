# Sarcasm Detection Project Progress

Hello everyone,

For my progress in the project, I first tried to build a pipeline using a Large Language Model (LLM) for sarcasm detection, leveraging the Riloff Tweet dataset from the paper I shared. You can find the dataset using this link: [Riloff Tweet Dataset](https://github.com/MirunaPislar/Sarcasm-Detection/blob/master/res/README.md).

## Step 1: Initial LLM Model

As a starting point, I used the test dataset (588 samples) with a simple prompt to classify the input as either sarcastic or non-sarcastic:

**This is a sarcasm classification task. Determine whether the following input text expresses sarcasm. Input: {input} If it does, output 'sarcastic'; otherwise, output 'non-sarcastic'.**

I got the following evaluation metrics:

- **Accuracy**: 0.43
- **Precision**: 0.06
- **Recall**: 0.17
- **F1 Score**: 0.09

## Step 2: Few-Shot Examples

Next, I improved the prompt by using a technique called Few-Shot Examples from the trained samples. The prompt template is as follows:

**This is a sarcasm classification task. Determine whether the following input text expresses sarcasm.

Here are some examples:

Input: Absolutely love when water is spilled on my phone.. Just love it.. #timeforanewphone Output: non-sarcastic

Input: I was hoping just a LITTLE more shit could hit the fan this week. Output: non-sarcastic

Input: @pdomo Don't forget that Nick Foles is also the new Tom Brady. What a preseason! #toomanystudQBs #thankgodwedonthavetebow Output: sarcastic

Input: I constantly see tweets about Arsenal on Twitter. Thanks for keeping the world updated @ZachBaugus & @shawnxh. #HugeArsenalFans Output: sarcastic

Now, classify the following input text:

Input: {input}

If it expresses sarcasm, output 'sarcastic'; otherwise, output 'non-sarcastic'.**

This led to a slight improvement in accuracy:

- **Accuracy**: 0.45
- **Precision**: 0.05
- **Recall**: 0.14
- **F1 Score**: 0.07

## Step 3: Retrieval-Augmented Generation (RAG)

Finally, I implemented the Retrieval-Augmented Generation (RAG) process, where I store the trained data as external knowledge (in a vector database) to aid the LLM in classification. After testing RAG with the same test data (data that the LLM hadn't seen before), the accuracy improved significantly to:

- **Accuracy**: 0.84

For all the details, I've uploaded the source code along with the data on GitHub.

---

Thank you for following along with the progress of this project!
