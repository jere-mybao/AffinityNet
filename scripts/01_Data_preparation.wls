#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Set the working directory *)
SetDirectory["~/repos/InhibitGPT/scripts"];

(* Import the dataset *)
d = Import["../data/BACE-1.xlsx", "Dataset", HeaderLines -> 1];
If[d === $Failed, Print["Failed to load data. Check the file path."]; Quit[]];
d = d[[1]];

(* Create a histogram of pIC50 values *)
Histogram[
  Normal@d[All, "pIC50"],
  Frame -> True, FrameLabel -> {"pIC50", "Count"}] // 
    Export["../figures/pIC50_distribution.jpg", #] &;

(* Create a visualization of the active vs. inactive classes based on the 'Class' column *)
PieChart[
  Values@Counts[Map[If[#["Class"] == 1, "Active", "Inactive"] &, d]],
  ChartLabels -> {"Active", "Inactive"},
  ChartStyle -> {Green, Red}] //
    Export["../figures/active_inactive_classes.jpg", #] &;

(* Helper function to format and export data *)
exportData[model_] := Module[{filteredData, formattedData},
   (* Filter rows based on the 'Model' column *)
   filteredData = Select[d, #["Model"] == model &];
   
   (* Format data as mol -> activity *)
   formattedData = Map[
     Function[row,
       row["mol"] -> If[row["Class"] == 1, "Active", "Inactive"]
     ],
     Normal[filteredData]
   ];

   (* Export formatted data to a text file *)
   Export[
     StringTemplate["../data/`model`.txt"]@<|"model" -> ToLowerCase[model]|>,
     StringJoin[
       Map[
         Function[pair, StringTemplate["\"`mol`\" -> \"`class`\"\n"][
           <|"mol" -> First[pair], "class" -> Last[pair]|>
         ]],
         formattedData
       ]
     ],
     "Text"
   ];
   
   (* Return the formatted data for .mx saving *)
   formattedData
];

(* Generate and save train and test files *)
train = exportData["Train"];
test = exportData["Valid"];

(* Save train and test datasets to a .mx file *)
DumpSave["../data/split.mx", {train, test}];

Print["train.txt, test.txt, and split.mx have been created successfully!"];
