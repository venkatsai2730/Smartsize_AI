'use client';

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Card } from "@/components/ui/card";
import { Loader2 } from "lucide-react";
import axios from "axios";

export default function ProcessingPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const measurementId = searchParams.get("measurement_id");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!measurementId) {
      setError("No measurement ID provided.");
      return;
    }

    const pollStatus = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/api/v1/measurements/${measurementId}`);
        if (response.data.chest) { // Assuming chest is set when processing is complete
          router.push(`/results?measurement_id=${measurementId}`);
        } else {
          setTimeout(pollStatus, 2000); // Poll every 2 seconds
        }
      } catch (err) {
        setError("Failed to check processing status. Please try again.");
      }
    };

    pollStatus();
  }, [measurementId, router]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md text-center animate-fade-in">
        <div className="p-8">
          <Loader2 className="h-12 w-12 mx-auto mb-4 animate-spin text-primary" />
          <h2 className="text-2xl font-bold mb-2">We’re Scanning Your Fit!</h2>
          <p className="text-foreground/80">
            Please wait while we process your measurements...
          </p>
          {error && <p className="text-red-500 mt-4">{error}</p>}
        </div>
      </Card>
    </div>
  );
}