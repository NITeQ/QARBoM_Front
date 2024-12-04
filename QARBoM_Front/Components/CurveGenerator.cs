using ScottPlot;
using ScottPlot.Plottables;
using ScottPlot.WPF;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace QARBoM_Front.Components
{
    internal class CurveGenerator
    {
        public CurveGenerator(int epochs, List<string> metricsList, WpfPlot mainPlot) 
        {
            EpochNumber = epochs;
            MetricsList = metricsList;
            MainPlot = mainPlot;

            MainPlot.Plot.Clear();
            GenerateSignals();
        }

        ScottPlot.Plottables.Signal? MSECurve;
        ScottPlot.Plottables.Signal? AccuracyCurve;
        ScottPlot.Plottables.Signal? PrecisionCurve;
        ScottPlot.Plottables.Signal? RecallCurve;

        private double[]? MSEList;
        private double[]? AccuracyList;
        private double[]? PrecisionList;
        private double[]? RecallList;

        private int MSEIndex = 0;
        private int AccuracyIndex = 0;
        private int PrecisionIndex = 0;
        private int RecallIndex = 0;

        private int EpochNumber;
        private List<string> MetricsList;
        private WpfPlot MainPlot; 

        private void GenerateSignals()
        {
            foreach (string metric in MetricsList)
            {
                switch (metric)
                {
                    case "Accuracy":
                        AccuracyList = new double[EpochNumber];
                        AccuracyCurve = MainPlot.Plot.Add.Signal(AccuracyList);
                        AccuracyCurve.LegendText = "Accuracy";
                        AccuracyCurve.LineColor = Colors.IndianRed;
                        AccuracyCurve.MarkerColor = Colors.IndianRed;
                        AccuracyCurve.LineWidth = 3;
                        break;
                    case "MSE":
                        MSEList = new double[EpochNumber];
                        MSECurve = MainPlot.Plot.Add.Signal(MSEList);
                        break;
                    case "Precision":
                        PrecisionList = new double[EpochNumber];
                        PrecisionCurve = MainPlot.Plot.Add.Signal(PrecisionList);
                        PrecisionCurve.LegendText = "Precision";
                        PrecisionCurve.LineColor = Colors.GoldenRod;
                        PrecisionCurve.MarkerColor = Colors.GoldenRod;
                        PrecisionCurve.LineWidth = 3;
                        break;
                    case "Recall":
                        RecallList = new double[EpochNumber];
                        RecallCurve = MainPlot.Plot.Add.Signal(RecallList);
                        RecallCurve.LegendText = "Recall";
                        RecallCurve.LineColor = Colors.MediumPurple;
                        RecallCurve.MarkerColor = Colors.MediumPurple;
                        RecallCurve.LineWidth = 3;
                        break;

                }
            }

            MainPlot.Plot.Axes.SetLimits(0, 5, 0, 1.5);
            MainPlot.Refresh();
        }

        private void UpdatePlotAuto(Signal? signal, int index)
        {

            //await new Task(() =>
            //{
            //    if (signal == null) return;

            //    signal.MaxRenderIndex = index;

            //    MainPlot.Refresh();
            //    MainPlot.Plot.Axes.AutoScale();
            //});

            if (signal == null) return;

            signal.MaxRenderIndex = index;

            MainPlot.Refresh();
            MainPlot.Plot.Axes.AutoScale();


        }

        public void UpdateListAuto(string data, bool updatePlot = true)
        {
            if (string.IsNullOrEmpty(data)) return;

            if (data.Contains("mse") && MSEIndex < EpochNumber)
            {
                data = data.Remove(0, 5);

                if (double.TryParse(data, out double value))
                {
                    if (MSEList == null)
                    {
                        MSEList = new double[EpochNumber];
                        MSEList[MSEIndex] = value;
                        MSEIndex++;
                    }
                    else
                    {
                        MSEList[MSEIndex] = value;
                        MSEIndex++;
                    }
                }
            }
            else if (data.Contains("accuracy") && AccuracyIndex < EpochNumber)
            {
                data = data.Split(":")[1].Substring(1);


                if (double.TryParse(data, out double value))
                {
                    if (AccuracyList == null)
                    {
                        AccuracyList = new double[EpochNumber];
                        AccuracyList[AccuracyIndex] = value;

                        AccuracyIndex++;
                    }
                    else
                    {
                        AccuracyList[AccuracyIndex] = value;
                        if (updatePlot) UpdatePlotAuto(AccuracyCurve, AccuracyIndex);


                        AccuracyIndex++;
                    }
                }
            }
            else if (data.Contains("precision") && PrecisionIndex < EpochNumber)
            {
                data = data.Split(":")[1].Substring(1);

                if (double.TryParse(data, out double value))
                {
                    if (PrecisionList == null)
                    {
                        PrecisionList = new double[EpochNumber];
                        PrecisionList[PrecisionIndex] = value;

                        PrecisionIndex++;
                    }
                    else
                    {
                        PrecisionList[PrecisionIndex] = value;
                        if (updatePlot) UpdatePlotAuto(PrecisionCurve, PrecisionIndex);


                        PrecisionIndex++;
                    }
                }

            }
            else if (data.Contains("recall") && RecallIndex < EpochNumber)
            {
                data = data.Split(":")[1].Substring(1);

                if (double.TryParse(data, out double value))
                {
                    if (RecallList == null)
                    {
                        RecallList = new double[EpochNumber];
                        RecallList[RecallIndex] = value;

                        RecallIndex++;
                    }
                    else
                    {
                        RecallList[RecallIndex] = value;
                        if (updatePlot) UpdatePlotAuto(RecallCurve, RecallIndex);


                        RecallIndex++;
                    }
                }

            }

        }
        public void UpdateListManual(string listName, double newValue)
        {
            switch (listName)
            {
                case "Accuracy":
                    if (AccuracyList == null)
                    {
                        AccuracyList = [newValue];
                    }
                    else AccuracyList.Append(newValue);
                    break;
                case "Precision":
                    if (PrecisionList == null)
                    {
                        PrecisionList = [newValue];
                    }
                    else PrecisionList.Append(newValue);
                    break;
                case "Recall":
                    if (RecallList == null)
                    {
                        RecallList = [newValue];
                    }
                    else RecallList.Append(newValue);
                    break;
                case "MSE":
                    if (MSEList == null)
                    {
                        MSEList = [newValue];
                    }
                    else MSEList.Append(newValue);
                    break;
                default:
                    break;
            }
        }
        public List<double>? GetList(string listName)
        {
            switch (listName)
            {
                case "Accuracy":
                    return AccuracyList?.ToList();
                case "Precision":
                    return PrecisionList?.ToList();
                case "Recall":
                    return RecallList?.ToList();
                case "MSE":
                    return MSEList?.ToList();
                default:
                    return null;
                    
            }
        }

        public int GetIndex(string listName)
        {
            switch (listName)
            {
                case "Accuracy":
                    return AccuracyIndex;
                case "Precision":
                    return PrecisionIndex;
                case "Recall":
                    return RecallIndex;
                case "MSE":
                    return MSEIndex;
                default:
                    return -1;

            }
        }
        public List<ScottPlot.Plottables.Signal> GetPlotList()
        {
            List<Signal> signals = new List<Signal>();

            if (MSECurve != null) signals.Append(MSECurve);
            if (AccuracyCurve  != null) signals.Append(AccuracyCurve);
            if (PrecisionCurve != null) signals.Append(PrecisionCurve);
            if (RecallCurve != null) signals.Append(RecallCurve);


            return signals;
        }
        

    }

}
