'use client';

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import Webcam from "react-webcam";
import { Upload, Camera, Loader2 } from "lucide-react";
import axios from "axios";
import { useRouter } from "next/navigation";

export default function UploadPage() {
  const [frontImage, setFrontImage] = useState<File | null>(null);
  const [sideImage, setSideImage] = useState<File | null>(null);
  const [height, setHeight] = useState<string>("");
  const [isCameraOpen, setIsCameraOpen] = useState<"front" | "side" | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const webcamRef = useRef<Webcam>(null);
  const router = useRouter();

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>, type: "front" | "side") => {
    const file = e.target.files?.[0];
    if (file) {
      if (type === "front") setFrontImage(file);
      else setSideImage(file);
    }
  };

  const captureImage = (type: "front" | "side") => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      fetch(imageSrc)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], `${type}.jpg`, { type: "image/jpeg" });
          if (type === "front") setFrontImage(file);
          else setSideImage(file);
          setIsCameraOpen(null);
        });
    }
  };

  const handleSubmit = async () => {
    if (!frontImage || !sideImage) {
      setError("Please upload both front and side images.");
      return;
    }
    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("front_image", frontImage);
    formData.append("side_image", sideImage);
    if (height) formData.append("height", height);

    try {
      const response = await axios.post("http://localhost:8000/api/v1/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      router.push(`/processing?measurement_id=${response.data.measurement_id}`);
    } catch (err) {
      setError("Failed to upload images. Please try again.");
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-2xl animate-fade-in">
        <h2 className="text-3xl font-bold text-center mb-6">Upload Your Photos</h2>
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium mb-2">Front Image</label>
            {isCameraOpen === "front" ? (
              <div className="space-y-4">
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  className="w-full rounded-md"
                />
                <div className="flex gap-4">
                  <Button onClick={() => captureImage("front")} className="btn-primary">
                    Capture
                  </Button>
                  <Button onClick={() => setIsCameraOpen(null)} variant="outline">
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4">
                {frontImage && (
                  <img
                    src={URL.createObjectURL(frontImage)}
                    alt="Front"
                    className="w-32 h-32 object-cover rounded-md"
                  />
                )}
                <div className="flex gap-4">
                  <Button onClick={() => setIsCameraOpen("front")} className="btn-secondary">
                    <Camera className="mr-2 h-4 w-4" /> Use Camera
                  </Button>
                  <label className="btn btn-primary cursor-pointer">
                    <Upload className="mr-2 h-4 w-4" /> Upload File
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => handleFileUpload(e, "front")}
                    />
                  </label>
                </div>
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Side Image</label>
            {isCameraOpen === "side" ? (
              <div className="space-y-4">
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  className="w-full rounded-md"
                />
                <div className="flex gap-4">
                  <Button onClick={() => captureImage("side")} className="btn-primary">
                    Capture
                  </Button>
                  <Button onClick={() => setIsCameraOpen(null)} variant="outline">
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4">
                {sideImage && (
                  <img
                    src={URL.createObjectURL(sideImage)}
                    alt="Side"
                    className="w-32 h-32 object-cover rounded-md"
                  />
                )}
                <div className="flex gap-4">
                  <Button onClick={() => setIsCameraOpen("side")} className="btn-secondary">
                    <Camera className="mr-2 h-4 w-4" /> Use Camera
                  </Button>
                  <label className="btn btn-primary cursor-pointer">
                    <Upload className="mr-2 h-4 w-4" /> Upload File
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => handleFileUpload(e, "side")}
                    />
                  </label>
                </div>
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Height (meters)</label>
            <Input
              type="number"
              step="0.01"
              placeholder="e.g., 1.75"
              value={height}
              onChange={(e) => setHeight(e.target.value)}
              className="w-full"
            />
          </div>

          {error && <p className="text-red-500 text-center">{error}</p>}

          <Button
            onClick={handleSubmit}
            className="btn-primary w-full"
            disabled={isUploading}
          >
            {isUploading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Uploading...
              </>
            ) : (
              "Submit"
            )}
          </Button>
        </div>
      </Card>
    </div>
  );
}