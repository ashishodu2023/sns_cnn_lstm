# analysis/evaluation.py
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
)
from matplotlib.backends.backend_pdf import PdfPages

class ModelEvaluator:
    """
    Provides evaluation metrics and plots:
      - Confusion Matrix, ROC Curve, Precision-Recall Curve
      - Calculates accuracy, precision, recall, F1, ROC AUC, and PR AUC.
      - Exports all plots and a metrics summary into a single PDF.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate_model(self, model, X_test_trace, X_test_scalar, y_test, threshold=0.5):
        self.logger.info("Evaluating the model...")
        y_pred_prob = model.predict([X_test_trace, X_test_scalar]).ravel()
        y_pred_class = (y_pred_prob > threshold).astype(int)
        report = classification_report(y_test, y_pred_class)
        self.logger.info("Classification Report:\n%s", report)
        cm = confusion_matrix(y_test, y_pred_class)
        self.logger.info("Confusion Matrix:\n%s", cm)
        return y_pred_prob, y_pred_class, cm

    def plot_confusion_matrix(self, cm, save_path=None):
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        if save_path:
            plt.savefig(save_path, format='pdf')
            plt.close()
            self.logger.info("Confusion matrix plot saved to %s", save_path)
        else:
            plt.show()

    def plot_roc_curve(self, y_test, y_pred_prob, save_path=None):
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc_val = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        if save_path:
            plt.savefig(save_path, format='pdf')
            plt.close()
            self.logger.info("ROC curve plot saved to %s", save_path)
        else:
            plt.show()

    def plot_precision_recall_curve(self, y_test, y_pred_prob, save_path=None):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc_val = auc(recall, precision)
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc_val:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        if save_path:
            plt.savefig(save_path, format='pdf')
            plt.close()
            self.logger.info("Precision-Recall curve plot saved to %s", save_path)
        else:
            plt.show()

    def export_metrics_pdf(self, cm, y_test, y_pred_prob, y_pred_class, pdf_filename='metrics_report.pdf'):
        self.logger.info("Exporting metrics and plots to PDF: %s", pdf_filename)
        TN, FP, FN, TP = cm.ravel()
        accuracy_val = (TP + TN) / np.sum(cm)
        TPR_val = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR_val = FP / (FP + TN) if (FP + TN) > 0 else 0
        precision_val = precision_score(y_test, y_pred_class)
        recall_val = recall_score(y_test, y_pred_class)
        f1_val = f1_score(y_test, y_pred_class)

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
            ax.plot(fpr_curve, tpr_curve, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right")
            pdf.savefig(fig)
            plt.close(fig)

            # Page 3: Precision-Recall Curve
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(rec_curve, prec_curve, label=f'PR Curve (AUC = {pr_auc_val:.2f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title("Precision-Recall Curve")
            ax.legend(loc="lower left")
            pdf.savefig(fig)
            plt.close(fig)

            # Page 4: Metrics Summary
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            summary_text = (
                f"Accuracy: {accuracy_val:.2f}\n"
                f"TPR (Recall): {TPR_val:.2f}\n"
                f"FPR: {FPR_val:.2f}\n"
                f"Precision: {precision_val:.2f}\n"
                f"Recall: {recall_val:.2f}\n"
                f"F1 Score: {f1_val:.2f}\n"
                f"ROC AUC: {roc_auc_val:.2f}\n"
                f"PR AUC: {pr_auc_val:.2f}\n"
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

        self.logger.info("Metrics exported as PDF: %s", pdf_filename)
