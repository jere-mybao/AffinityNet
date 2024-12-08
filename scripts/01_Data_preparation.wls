#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Set the working directory (update the path as needed) *)
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

(* Filter and export data by 'Model' column with custom formatting *)
exportData[model_] := Module[{filteredData, formattedData},
   (* Filter rows based on the 'Model' column *)
   filteredData = Select[d, #["Model"] == model &];
   
   (* Format data as "mol" -> "Active"/"Inactive" *)
    formattedData = StringJoin[
      Map[
        Function[row, 
          StringTemplate["\"`mol`\" -> \"`class`\"\n"][
            <|
              "mol" -> row["mol"], 
              "class" -> If[row["Class"] == 1, "Active", "Inactive"]
            |>
          ]
        ],
        Normal[filteredData] (* Ensure the data is a list, not a Dataset *)
      ]
    ];

   (* Export formatted data to a text file *)
   Export[
     StringTemplate["../data/`model`.txt"]@<|"model" -> ToLowerCase[model]|>,
     formattedData,
     "Text"
   ];
];

(* Export Train and Valid files *)
exportData["Train"];
exportData["Valid"];
