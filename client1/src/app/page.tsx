'use client';

import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-primary to-background p-4">
      <div className="text-center animate-fade-in">
        <h1 className="text-5xl font-bold text-[#FFF7ED] mb-4">SmartSize AI</h1>
        <p className="text-xl text-[#E5E7EB] mb-8 max-w-md mx-auto">
          Get precise body measurements from a photo. Perfect fit, every time.
        </p>
        <Button
          className="btn-primary text-lg"
          onClick={() => router.push("/upload")}
        >
          Start Measuring <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
      </div>
    </div>
  );
}