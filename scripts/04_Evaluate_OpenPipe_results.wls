#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Set the working directory *)
SetDirectory["~/repos/InhibitGPT/scripts"];

(* Load necessary functions *)
<<"../src/finetuning.wl";

(* Import the JSONL results file *)
results = importJSONL["../results/BACE1_Molecules_qwen25_(test_entries).jsonl"];

(* Process results into an association *)
outputs = AssociationThread[
  results[[1, "outputs", All, "model"]],
  Transpose[results[[All, "outputs", All, "message", "content"]]]
];

(* Check for the key existence and handle missing keys *)
If[!KeyExistsQ[outputs, "openpipe:molecules-qwen-25"],
  Print["Key 'openpipe:molecules-qwen-25' is missing. Available keys: ", Keys[outputs]];
  Quit[]
];

(* Generate classifier measurements *)
cm = ClassifierMeasurements[
  outputs["openpipe:molecules-qwen-25"],
  outputs["Dataset Output"]
];

(* Enhance the ConfusionMatrixPlot with additional settings *)
confMatrix = cm;

(* Export the enhanced confusion matrix plot *)
Export["../figures/qwen25_predictions.jpg", confMatrix, "JPEG"];
Print["Enhanced confusion matrix plot saved successfully."];