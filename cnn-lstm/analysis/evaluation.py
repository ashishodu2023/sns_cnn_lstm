#!/usr/bin/env python
# coding: utf-8

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, precision_score, recall_score, f1_score
)
from matplotlib.backends.backend_pdf import PdfPages
from utils.logger import Logger

class ModelEvaluator:
    """
    Provides evaluation metrics and plots:
      - Confusion Matrix, ROC Curve, Precision-Recall Curve.
      - Calculates accuracy, TPR (recall), FPR, precision, specificity, F1 score, and MCC.
      - Exports all plots and a metrics summary into a single PDF.
    """
    def __init__(self):
        self.logger = Logger()

    def evaluate_model(self, model, X_test_trace, X_test_scalar, y_test, threshold=0.5):
        self.logger.info("Evaluating the model...")
        y_pred_prob = model.predict([X_test_trace, X_test_scalar]).ravel()
        y_pred_class = (y_pred_prob > threshold).astype(int)
        report = classification_report(y_test, y_pred_class)
        self.logger.info(f"Classification Report:\n{report}")
        cm = confusion_matrix(y_test, y_pred_class)
        self.logger.info("Confusion Matrix:\n {cm}")
        return y_pred_prob, y_pred_class, cm

    def plot_confusion_matrix(self, cm, title="Confusion Matrix"):
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def plot_roc_curve(self, y_test, y_pred_prob):
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc_val = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_val:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self, y_test, y_pred_prob):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc_val = auc(recall, precision)
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc_val:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.show()

    def export_metrics_pdf(self, cm, y_test, y_pred_prob, y_pred_class, pdf_filename='metrics_report.pdf'):
        self.logger.info(f"Exporting metrics and plots to PDF:{ pdf_filename}")
        
        # Calculate metrics
        TN, FP, FN, TP = cm.ravel()
        total = TN + FP + FN + TP
        accuracy = (TP + TN) / total
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        precision_val = TP / (TP + FP) if (TP + FP) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        F1 = 2 * precision_val * TPR / (precision_val + TPR) if (precision_val + TPR) > 0 else 0
        mcc_denom = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
        MCC = (TP * TN - FP * FN) / mcc_denom if mcc_denom != 0 else 0

        # Calculate ROC curve and PR curve values
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_prob)
        roc_auc_val = auc(fpr_curve, tpr_curve)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc_val = auc(rec_curve, prec_curve)

        class_report = classification_report(y_test, y_pred_class)

        with PdfPages(pdf_filename) as pdf:
            # Page 1: Confusion Matrix
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Confusion Matrix")
            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: ROC Curve
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr_curve, tpr_curve, label=f"ROC curve (AUC = {roc_auc_val:.2f})")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right")
            pdf.savefig(fig)
            plt.close(fig)

            # Page 3: Precision-Recall Curve
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(rec_curve, prec_curve, label=f"PR curve (AUC = {pr_auc_val:.2f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            ax.legend(loc="lower left")
            pdf.savefig(fig)
            plt.close(fig)

            # Page 4: Metrics Summary
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            summary_text = (
                f"Accuracy: {accuracy:.4f}\n"
                f"TPR (Recall): {TPR:.4f}\n"
                f"FPR: {FPR:.5f}\n"
                f"Precision: {precision_val:.4f}\n"
                f"Specificity: {specificity:.4f}\n"
                f"F1 Score: {F1:.4f}\n"
                f"MCC: {MCC:.4f}\n"
            )
            ax.text(0.1, 0.5, summary_text, fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)

            # Page 5: Classification Report
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            ax.text(0.05, 0.95, "Classification Report", fontsize=16, weight='bold', transform=ax.transAxes)
            ax.text(0.05, 0.90, class_report, fontsize=12, transform=ax.transAxes)
            pdf.savefig(fig)
            plt.close(fig)

        self.logger.info(f"Metrics exported as PDF:{pdf_filename}")