using System;
using System.Collections.Generic;
using System.Drawing.Text;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace QARBoM_Front.Components
{
    internal class EpochResults : IComparable<EpochResults>, IEquatable<EpochResults>
    {
        public EpochResults() { }

        private int _epochNumber;
        private double _accuracy;
        private double _precision;
        private double _recall;
        private double _f1;

        private int _falseNegatives;
        private int _trueNegatives;
        private int _falsePositives;
        private int _truePositives;

        public int EpochNumber { get { return _epochNumber; } set { _epochNumber = value; } }
        public double Accuracy { get { return _accuracy; } set { _accuracy = value; } }
        public double Precision { get { return _precision; } set { _precision = value; } }
        public double Recall { get { return _recall; } set { _recall = value; } }
        public double F1 { get { return _f1; } set { _f1 = value; } }
        public int FalseNegatives { get { return _falseNegatives; } set { _falseNegatives = value; } }
        public int TrueNegatives { get { return _trueNegatives; } set { _trueNegatives = value; } }
        public int FalsePositives { get { return _falsePositives; } set { _falsePositives = value; } }
        public int TruePositives { get { return _truePositives; } set { _truePositives = value; } }

        public int CompareTo(EpochResults? other)
        {
            if (other == null) throw new ArgumentException("Incorrect type for comparison");


            if (this.Accuracy > other.Accuracy) return 1;
            else if (this.Accuracy < other.Accuracy) return -1;
            else return 0;
            
        }

        public bool Equals(EpochResults? other)
        {
            if (other == null) throw new ArgumentException("Incorrect type for comparison");
            return this.EpochNumber == other.EpochNumber;
        }
    }

}
