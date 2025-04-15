'use client';

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Download } from "lucide-react";
import axios from "axios";
import jsPDF from "jspdf";
import { cn } from "@/lib/utils";

interface Measurement {
  id: number;
  user_id: string;
  chest: number | null;
  waist: number | null;
  hips: number | null;
  shoulder_width: number | null;
  arm_length: number | null;
  leg_length: number | null;
  inseam: number | null;
  neck: number | null;
  created_at: string;
}

export default function ResultsPage() {
  const searchParams = useSearchParams();
  const measurementId = searchParams.get("measurement_id");
  const [measurement, setMeasurement] = useState<Measurement | null>(null);
  const [size, setSize] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!measurementId) {
      setError("No measurement ID provided.");
      return;
    }

    const fetchResults = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/api/v1/measurements/${measurementId}`);
        setMeasurement(response.data);

        // Mock size recommendation (since backend doesn't return size directly)
        const chest = response.data.chest || 85;
        const sizeChart: { [key: string]: [number, number] } = {
          XS: [65, 80],
          S: [80, 90],
          M: [90, 100],
          L: [100, 110],
          XL: [110, 120],
          XXL: [120, 135],
        };
        let recommendedSize = "Unknown";
        for (const [size, [min, max]] of Object.entries(sizeChart)) {
          if (chest >= min && chest < max) {
            recommendedSize = size;
            break;
          }
        }
        if (chest >= 135) recommendedSize = "XXXL";
        else if (chest < 65) recommendedSize = "XS";
        setSize(recommendedSize);
      } catch (err) {
        setError("Failed to load results. Please try again.");
      }
    };

    fetchResults();
  }, [measurementId]);

  const downloadPDF = () => {
    if (!measurement) return;
    const doc = new jsPDF();
    doc.setFontSize(18);
    doc.text("SmartSize AI Measurement Report", 20, 20);
    doc.setFontSize(12);
    doc.text(`Date: ${new Date(measurement.created_at).toLocaleDateString()}`, 20, 30);
    doc.text(`Recommended Size: ${size || "Unknown"}`, 20, 40);
    doc.text("Measurements (cm):", 20, 50);
    let y = 60;
    const fields = [
      "chest",
      "waist",
      "hips",
      "shoulder_width",
      "arm_length",
      "leg_length",
      "inseam",
      "neck",
    ];
    fields.forEach((field) => {
      const value = measurement[field as keyof Measurement];
      if (typeof value === "number") {
        doc.text(`${field.replace("_", " ").toUpperCase()}: ${value.toFixed(2)} cm`, 20, y);
        y += 10;
      }
    });
    doc.save("smartsize-measurements.pdf");
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#F8FAFC] p-4">
      <Card className="w-full max-w-3xl animate-fade-in">
        <div className="p-8">
          <h2 className="text-3xl font-bold text-center mb-6">Your Measurements</h2>
          {error ? (
            <p className="text-red-500 text-center">{error}</p>
          ) : !measurement ? (
            <p className="text-center">Loading results...</p>
          ) : (
            <div className="space-y-6">
              <div className="text-center">
                <h3 className="text-xl font-semibold">Recommended Size</h3>
                <p className="text-2xl text-primary">{size || "Unknown"}</p>
              </div>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Measurement</TableHead>
                    <TableHead className="text-right">Value (cm)</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[
                    { key: "chest", label: "Chest" },
                    { key: "waist", label: "Waist" },
                    { key: "hips", label: "Hips" },
                    { key: "shoulder_width", label: "Shoulder Width" },
                    { key: "arm_length", label: "Arm Length" },
                    { key: "leg_length", label: "Leg Length" },
                    { key: "inseam", label: "Inseam" },
                    { key: "neck", label: "Neck" },
                  ].map(({ key, label }) => {
                    const value = measurement[key as keyof Measurement];
                    return (
                      <TableRow key={key}>
                        <TableCell>{label}</TableCell>
                        <TableCell className="text-right">
                          {typeof value === "number" ? value.toFixed(2) : "N/A"}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
              <div className="flex justify-center">
                <Button onClick={downloadPDF} className="btn-primary">
                  <Download className="mr-2 h-4 w-4" /> Download as PDF
                </Button>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}