#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Set the working directory *)
SetDirectory["~/repos/InhibitGPT/scripts"];

<<"../src/finetuning.wl"
<<"../data/split.mx"


system = "You are a cheminformatics expert predicting the activity state of a molecule based on its chemical structure. Answer only with the following activity states and no other output: Active, Inactive.";

tr = formatChatML[system, #]&/@train;
te = formatChatML[system, #]&/@test;


exportChatML["../data/train.jsonl", tr]
exportChatML["../data/test.jsonl", te]

exportOpenPipeChatML["../data/openpipe.jsonl", tr, te]

(* after this, upload the file to Openpipe.ai and initiate fine tunings...*)



