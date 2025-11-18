Perfect 👍 — I’ll turn your Markdown into a **LaTeX thesis-ready template**.
This will look professional in your report, with sections, equations, and a figure placeholder for Grad-CAM.

---

```latex
\section{Interactive Decision Support Layer for Melanoma Classification}

We extend a convolutional neural network (CNN) classifier with an \textit{interactive decision support} layer that enhances transparency and robustness.  
The system integrates Grad-CAM explanations, detects uncertainty, and adaptively asks the user 2--4 targeted follow-up questions.  
Responses are fused with the image model’s output using a lightweight calibration model, yielding an updated melanoma probability.

\subsection{System Components}
\begin{itemize}
    \item \textbf{Core Model:} ResNet50 (or stronger) trained on dermoscopic images with lesion labels.  
    \item \textbf{Uncertainty Gate:} Detects low confidence based on probability, entropy, or class margin.  
    \item \textbf{Question Policy:} A fixed set of clinically relevant yes/no or categorical questions.  
    \item \textbf{Calibration Model:} Logistic regression combining image logits with encoded answers.  
    \item \textbf{Explanation Layer:} Grad-CAM heatmaps with textual risk factor commentary.  
\end{itemize}

\subsection{Uncertainty Estimation}
Confidence is quantified by multiple metrics:
\begin{align}
    \text{Max Prob: } & p_{\max} = \max_i p(y=i \mid x) \\
    \text{Entropy: } & H = - \sum_i p_i \log p_i \\
    \text{Margin: } & \Delta = p_{\text{top}} - p_{\text{second}}
\end{align}

We trigger follow-up questions if $0.35 \leq p_{\max} \leq 0.75$,  
$H > 0.8$, or $\Delta < 0.2$.

\subsection{Question Set}
A small set of risk-related questions is used:
\begin{enumerate}
    \item Age group: $<$30, 30--50, $>$50  
    \item Lesion change: growth or evolution? (yes/no)  
    \item Sun exposure: frequent intense exposure? (yes/no)  
    \item Family history of melanoma (yes/no)  
    \item Skin phototype: I--II, III--IV, V--VI  
    \item Lesion location: trunk, head/neck, extremities  
\end{enumerate}

\subsection{Calibration Model}
We fit a logistic regression:
\begin{equation}
    P(\text{melanoma}) = \sigma \big( \beta_0 + \beta_1 \cdot \text{logit}_{img} + \sum_j w_j f_j \big)
\end{equation}
where $\text{logit}_{img}$ is the CNN score, and $f_j$ are encoded question responses.

\subsection{Interactive Update Procedure}
\begin{enumerate}
    \item Compute CNN probability $p_{img}$.  
    \item If uncertain, ask question $q_1$.  
    \item Encode answer and recompute $p_1$.  
    \item Continue for up to 3 questions or until confidence threshold reached.  
\end{enumerate}

\subsection{Explanation Layer}
Outputs consist of:
\begin{itemize}
    \item \textbf{Visual:} Grad-CAM overlay highlighting focus regions.  
    \item \textbf{Text:} Example: \textit{``Model attended to asymmetric regions; recent growth reported (increases risk); probability updated to 0.74.''}  
\end{itemize}

\subsection{Evaluation Metrics}
\begin{itemize}
    \item ROC-AUC, PR-AUC  
    \item Expected Calibration Error (ECE)  
    \item Sensitivity at 90\% specificity  
    \item Decision curve analysis (net benefit)  
\end{itemize}

\subsection{XAI Sanity Checks}
\begin{itemize}
    \item Randomization test: CAMs degrade under weight shuffling.  
    \item Insertion/deletion: probability drops when occluding CAM regions.  
    \item Overlap with lesion masks where available.  
\end{itemize}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.6\linewidth]{gradcam_example.png}
    \caption{Grad-CAM explanation overlay for a melanoma prediction.}
\end{figure}

\subsection{Ethical Guardrails}
\begin{itemize}
    \item Bias checks across demographics.  
    \item No storage of user-provided data without consent.  
    \item Disclaimer: \textit{``This tool is for research and educational purposes only, not medical diagnosis.''}  
\end{itemize}
```

---

⚡ This is **ready to drop into your thesis**.

Do you want me to also make a **Python notebook cell** that outputs results (Grad-CAM + updated probability + explanation text) in a **LaTeX-friendly table/figure** format, so you can directly include experimental results?
