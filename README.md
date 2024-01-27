# AutoMathText
AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts (https://huggingface.co/datasets/math-ai/AutoMathText).

**AutoMathText** is an extensive and carefully curated dataset encompassing **200 GB** of mathematical texts. It's a unique compilation sourced from a diverse range of platforms including various websites, arXiv, and GitHub (OpenWebMath, RedPajama, Algebraic Stack). This rich repository has been **autonomously selected (labeled) by the state-of-the-art open-sourced language model**, Qwen-72B. Each piece of content in the dataset is assigned **a score `lm_q1q2_score` within the range of [0, 1]**, reflecting its relevance, quality and educational value in the context of mathematical intelligence.

## Objective

The primary aim of the **AutoMathText** dataset is to provide a comprehensive and reliable resource for a wide array of users - from academic researchers and educators to AI practitioners and mathematics enthusiasts. This dataset is particularly geared towards:

- Facilitating advanced research in **the intersection of mathematics and artificial intelligence**.
- Serving as an educational tool for **learning and teaching complex mathematical concepts**.
- Providing **a foundation for developing and training AI models** specialized in processing and understanding **mathematical content**.

## Features

- **Volume**: Approximately 200 GB of text data (in natural language and programming language).
- **Content**: A diverse collection of mathematical texts, including but not limited to research papers, educational articles, and code documentation.
- **Labeling**: Every text is **scored** by Qwen-72B, a sophisticated language model, ensuring a high standard of relevance and accuracy.
- **Scope**: Covers a wide spectrum of mathematical topics, making it suitable for various applications in advanced research and education.

## References

- OpenWebMath [[link]](https://huggingface.co/datasets/open-web-math/open-web-math)
- RedPajama [[link]](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
- Algebraick Stack [[link]](https://huggingface.co/datasets/EleutherAI/proof-pile-2) (a subset of Proof-Pile-2)

## Citation 
We appreciate your use of **AutoMathText** in your work. If you find this repository helpful, please consider citing it and star this repo. Feel free to contact zhangyif21@tsinghua.edu.cn or open an issue if you have any questions.

```bibtex
@misc{automathtext2024,
      title={AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts},
      author={Zhang, Yifan and Luo, Yifan},
      year={2024},
}
