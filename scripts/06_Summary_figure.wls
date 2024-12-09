#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Set the working directory *)
SetDirectory["~/repos/InhibitGPT/scripts"];

BarChart[
    {Labeled[81.7, "Fine-tuned gpt-4o-mini", Right],
     Labeled[83.4, "Fine-tuned Llama-3.1-8B", Right],
     Labeled[81.4, "Fine-tuned Mistral-Nemo-12B", Right],
     Labeled[83.4, "Fine-tuned Qwen-2.5-72B", Right],
     Labeled[75.9, "SentenceBERT nearest", Right],
     Labeled[77.3, "String Edit nearest", Right],
     Labeled[55.8, "Guess majority", Right]},
    BarOrigin -> Left,
    ChartStyle -> {
        RGBColor[0.3, 0.5, 0.8],   (* Muted Blue *)
        RGBColor[0.4, 0.7, 0.4],   (* Soft Green *)
        RGBColor[0.8, 0.6, 0.2],   (* Golden Yellow *)
        RGBColor[0.7, 0.4, 0.6],   (* Muted Purple *)
        RGBColor[0.6, 0.6, 0.6],   (* Neutral Gray *)
        RGBColor[0.5, 0.6, 0.8],   (* Light Blue *)
        RGBColor[0.7, 0.5, 0.3]    (* Warm Brown *)
    },
    Frame -> True,
    FrameLabel -> {None, "% Accuracy"},
    BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 12},
    PlotLabel -> Style["Model Accuracy Comparison", Bold, FontSize -> 14],
    ImageSize -> Large
] // Export["../figures/summary_barchart.jpg", #] &
