import { useState, useEffect, useRef } from "react";
import * as faceapi from "@vladmandic/face-api";
import Tesseract from "tesseract.js";
import UploadIcon from "./assets/upload.svg";
import Webcam from "react-webcam";

export default function App() {
  const [proofType, setProofType] = useState("");
  const [frontImg, setFrontImg] = useState(null);
  const [backImg, setBackImg] = useState(null);
  const [faceImg, setFaceImg] = useState(null);
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [facingMode, setFacingMode] = useState("user"); // "user" for front camera, "environment" for back camera
  const [capturedImage, setCapturedImage] = useState(null);
  const [similarity, setSimilarity] = useState(null);
  const [isComparing, setIsComparing] = useState(false);
  const [detectedFaces, setDetectedFaces] = useState([]);
  const [isDetectingFaces, setIsDetectingFaces] = useState(false);
  const webcamRef = useRef(null);
  const faceImageRef = useRef(null);

  useEffect(() => {
    const loadModels = async () => {
      const MODEL_URL = "/models";
      try {
        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
          faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
        ]);
        console.log("✅ All FaceAPI models loaded successfully");
      } catch (error) {
        console.error("❌ Error loading models:", error);
        alert("Failed to load face detection models. Please refresh the page.");
      }
    };
    loadModels();
  }, []);

  const handleFrontUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const imgURL = URL.createObjectURL(file);
      setFrontImg(imgURL);
      setFaceImg(null);
      setInfo(null);
      setLoading(true);
      await detectFace(imgURL);
      await extractInfo(imgURL);
      setLoading(false);
    }
  };

  const handleBackUpload = (e) => {
    const file = e.target.files[0];
    if (file) setBackImg(URL.createObjectURL(file));
  };

  const detectFace = async (imgURL) => {
    const img = await faceapi.fetchImage(imgURL);
    const detection = await faceapi
      .detectSingleFace(
        img,
        new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 })
      )
      .withFaceLandmarks();

    if (detection) {
      const { x, y, width, height } = detection.detection.box;
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      const marginX = width * 0.3;
      const marginY = height * 0.4;

      const sx = Math.max(0, x - marginX);
      const sy = Math.max(0, y - marginY);
      const sWidth = width + marginX * 2;
      const sHeight = height + marginY * 2;

      canvas.width = sWidth;
      canvas.height = sHeight;
      ctx.drawImage(img, sx, sy, sWidth, sHeight, 0, 0, sWidth, sHeight);
      setFaceImg(canvas.toDataURL("image/png"));
    } else {
      console.warn("⚠️ No face detected");
    }
  };

  const extractInfo = async (imgURL) => {
    try {
      const result = await Tesseract.recognize(imgURL, "eng");
      const text = result.data.text;
      const lines = text
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean);

      let name = "Not found";
      let dob = "Not found";
      let nationality = "Not found";
      let passportNumber = "Not found";
      let serialNumber = "Not found";
      let passportExpiry = "Not found";
      let additionalFields = {};

      if (proofType === "aadhaar") {
        const dobLineIndex = lines.findIndex((line) =>
          /\d{2}\/\d{2}\/\d{4}/.test(line)
        );
        if (dobLineIndex >= 0) {
          dob =
            lines[dobLineIndex].match(/\d{2}\/\d{2}\/\d{4}/)?.[0] ||
            "Not found";
          if (dobLineIndex > 0)
            name = lines[dobLineIndex - 1].replace(/[^a-zA-Z\s]/g, "").trim();
        }

        const aadhaarLine = lines.find((line) =>
          /\d{4}\s?\d{4}\s?\d{4}/.test(line)
        );
        if (aadhaarLine)
          nationality = aadhaarLine
            .replace(/\D/g, "")
            .replace(/(\d{4})(?=\d)/g, "$1 ");
      } else if (proofType === "pan") {
        const panLineIndex = lines.findIndex((line) =>
          /[A-Z]{5}[0-9]{4}[A-Z]{1}/i.test(line)
        );
        if (panLineIndex >= 0) {
          nationality = lines[panLineIndex].match(
            /[A-Z]{5}[0-9]{4}[A-Z]{1}/i
          )[0];

          const nameLineIndex = lines.findIndex(
            (line) =>
              line.toLowerCase().includes("name") &&
              !line.toLowerCase().includes("father")
          );
          if (nameLineIndex >= 0 && nameLineIndex + 1 < lines.length) {
            name = lines[nameLineIndex + 1].replace(/[^a-zA-Z\s]/g, "").trim();
          }
        }

        const dobLine = lines.find((line) => /\d{2}\/\d{2}\/\d{4}/.test(line));
        if (dobLine) dob = dobLine.match(/\d{2}\/\d{2}\/\d{4}/)[0];
      } else if (proofType === "other") {
        const dobLineIndex = lines.findIndex((line) =>
          /\d{2}\/\d{2}\/\d{4}/.test(line)
        );
        if (dobLineIndex >= 0) {
          dob =
            lines[dobLineIndex].match(/\d{2}\/\d{2}\/\d{4}/)?.[0] ||
            "Not found";
        }

        // Extract name by searching for lines containing "NAME"
        const nameLineIndex = lines.findIndex((line) =>
          line.toUpperCase().includes("NAME")
        );
        if (nameLineIndex >= 0) {
          name = lines[nameLineIndex].replace(/[^a-zA-Z\s]/g, "").trim();
          // If "NAME" is in the line, get the text after it
          if (name.toUpperCase().includes("NAME")) {
            const nameParts = name.split(/NAME/i);
            if (nameParts.length > 1) {
              name = nameParts[1].trim();
            }
          }
        }

        // Extract passport number
        // Common passport number patterns
        const passportPatterns = [
          /[A-Z]{2}\d{7}/, // Standard format: 2 letters + 7 digits
          /\d{9}/, // 9 digits
          /[A-Z]\d{8}/, // 1 letter + 8 digits
          /[A-Z]{3}\d{6}/, // 3 letters + 6 digits
          /\d{8,9}/, // 8 or 9 digits
          /[A-Z]{1,2}\d{6,8}/, // 1-2 letters + 6-8 digits
        ];

        // Also look for lines containing "PASSPORT" or "PASS NO" or "PASS#"
        const passportTermIndex = lines.findIndex(
          (line) =>
            line.toUpperCase().includes("PASSPORT") ||
            line.toUpperCase().includes("PASS NO") ||
            line.toUpperCase().includes("PASS#")
        );

        if (passportTermIndex >= 0) {
          const line = lines[passportTermIndex];
          // Try to find passport number in the same line
          for (const pattern of passportPatterns) {
            const match = line.match(pattern);
            if (match) {
              passportNumber = match[0];
              break;
            }
          }
          // If not found in same line, check next line
          if (
            passportNumber === "Not found" &&
            passportTermIndex + 1 < lines.length
          ) {
            const nextLine = lines[passportTermIndex + 1];
            for (const pattern of passportPatterns) {
              const match = nextLine.match(pattern);
              if (match) {
                passportNumber = match[0];
                break;
              }
            }
          }
        } else {
          // If no passport terms found, search for patterns in all lines
          for (const line of lines) {
            for (const pattern of passportPatterns) {
              const match = line.match(pattern);
              if (match) {
                passportNumber = match[0];
                break;
              }
            }
            if (passportNumber !== "Not found") break;
          }
        }

        // Extract serial number (could be various formats)
        const serialPatterns = [
          /[A-Z]{2}\d{7,8}/, // 2 letters + 7-8 digits
          /\d{10,12}/, // 10-12 digits
          /[A-Z]\d{9,10}/, // 1 letter + 9-10 digits
          /[A-Z]{3,4}\d{5,6}/, // 3-4 letters + 5-6 digits
          /\d{8,9}[A-Z]?\d{1,2}/, // 8-9 digits + optional letter + 1-2 digits
        ];

        // Look for serial number patterns
        for (const line of lines) {
          for (const pattern of serialPatterns) {
            const match = line.match(pattern);
            if (match) {
              serialNumber = match[0];
              break;
            }
          }
          if (serialNumber !== "Not found") break;
        }

        // Extract passport expiry date
        const expiryPatterns = [
          /\d{2}\/\d{2}\/\d{4}/, // DD/MM/YYYY
          /\d{2}-\d{2}-\d{4}/, // DD-MM-YYYY
          /\d{4}-\d{2}-\d{2}/, // YYYY-MM-DD
          /\d{2}\/\d{2}\/\d{2}/, // DD/MM/YY
          /\d{2}-\d{2}\/\d{4}/, // DD-MM/YYYY
          /(?:EXP|EXPIRY|EXPIRES|VALID\s*UNTIL|VALID\s*THRU)\s*[:\-]?\s*(\d{2}\/\d{2}\/\d{4})/i,
        ];

        // Look for expiry terms first
        const expiryTermIndex = lines.findIndex(
          (line) =>
            line.toUpperCase().includes("EXP") ||
            line.toUpperCase().includes("EXPIRY") ||
            line.toUpperCase().includes("EXPIRES") ||
            line.toUpperCase().includes("VALID UNTIL") ||
            line.toUpperCase().includes("VALID THRU")
        );

        if (expiryTermIndex >= 0) {
          const line = lines[expiryTermIndex];
          // Try to find expiry date in the same line
          for (const pattern of expiryPatterns) {
            const match = line.match(pattern);
            if (match) {
              passportExpiry = match[1] || match[0];
              break;
            }
          }
          // If not found in same line, check next line
          if (
            passportExpiry === "Not found" &&
            expiryTermIndex + 1 < lines.length
          ) {
            const nextLine = lines[expiryTermIndex + 1];
            for (const pattern of expiryPatterns) {
              const match = nextLine.match(pattern);
              if (match) {
                passportExpiry = match[1] || match[0];
                break;
              }
            }
          }
        } else {
          // If no expiry terms found, search for date patterns in all lines
          for (const line of lines) {
            for (const pattern of expiryPatterns) {
              const match = line.match(pattern);
              if (match) {
                passportExpiry = match[1] || match[0];
                break;
              }
            }
            if (passportExpiry !== "Not found") break;
          }
        }

        // Extract additional fields by looking for common ID card terms
        const additionalTerms = {
          GENDER: "gender",
          SEX: "gender",
          FATHER: "fatherName",
          MOTHER: "motherName",
          SPOUSE: "spouseName",
          "ISSUING AUTHORITY": "issuingAuthority",
          "PLACE OF BIRTH": "placeOfBirth",
          "DATE OF ISSUE": "dateOfIssue",
          "ID NO": "idNumber",
          "DOCUMENT NO": "documentNumber",
          "FILE NO": "fileNumber",
        };

        for (const [term, field] of Object.entries(additionalTerms)) {
          const termIndex = lines.findIndex((line) =>
            line.toUpperCase().includes(term)
          );
          if (termIndex >= 0) {
            const line = lines[termIndex];
            let value = "Not found";

            // Try to get value from same line after the term
            const termParts = line.split(new RegExp(term, "i"));
            if (termParts.length > 1) {
              value = termParts[1].trim().replace(/[^a-zA-Z0-9\s\/\-]/g, "");
            }
            // If no value found in same line, check next line
            else if (termIndex + 1 < lines.length) {
              value = lines[termIndex + 1]
                .trim()
                .replace(/[^a-zA-Z0-9\s\/\-]/g, "");
            }

            if (value && value.length > 1 && value.length < 100) {
              additionalFields[field] = value;
            }
          }
        }

        // Extract nationality by searching for country names or nationality terms
        const countryNames = [
          "INDIA",
          "USA",
          "UNITED STATES",
          "UK",
          "UNITED KINGDOM",
          "CANADA",
          "AUSTRALIA",
          "GERMANY",
          "FRANCE",
          "ITALY",
          "SPAIN",
          "JAPAN",
          "CHINA",
          "BRAZIL",
          "RUSSIA",
          "SWITZERLAND",
          "NORWAY",
          "SWEDEN",
          "DENMARK",
          "FINLAND",
          "NETHERLANDS",
          "BELGIUM",
          "AUSTRIA",
          "IRELAND",
          "NEW ZEALAND",
          "SINGAPORE",
          "MALAYSIA",
          "THAILAND",
          "SOUTH KOREA",
          "MEXICO",
          "ARGENTINA",
          "CHILE",
          "COLOMBIA",
          "VENEZUELA",
          "PERU",
          "EGYPT",
          "SOUTH AFRICA",
          "KENYA",
          "NIGERIA",
          "GHANA",
          "MOROCCO",
          "TURKEY",
          "SAUDI ARABIA",
          "UAE",
          "QATAR",
          "KUWAIT",
          "OMAN",
          "PAKISTAN",
          "BANGLADESH",
          "NEPAL",
          "SRI LANKA",
          "MYANMAR",
          "VIETNAM",
          "INDONESIA",
          "PHILIPPINES",
          "CAMBODIA",
          "LAOS",
        ];

        const nationalityTerms = [
          "NATIONALITY",
          "CITIZENSHIP",
          "CITIZEN",
          "NATIONAL",
          "ORIGIN",
          "DOMICILE",
          "RESIDENT",
          "BELONGS TO",
          "FROM",
        ];

        // First try to find nationality terms
        for (const term of nationalityTerms) {
          const termIndex = lines.findIndex((line) =>
            line.toUpperCase().includes(term)
          );
          if (termIndex >= 0) {
            // Check if there's text after the term
            const line = lines[termIndex];
            const termParts = line.split(new RegExp(term, "i"));
            if (termParts.length > 1) {
              const potentialNationality = termParts[1]
                .trim()
                .replace(/[^a-zA-Z\s]/g, "");
              if (potentialNationality) {
                nationality = potentialNationality;
                break;
              }
            }
            // Check next line if current line doesn't have nationality after term
            if (termIndex + 1 < lines.length) {
              const nextLine = lines[termIndex + 1]
                .replace(/[^a-zA-Z\s]/g, "")
                .trim();
              if (nextLine && nextLine.length > 1 && nextLine.length < 50) {
                nationality = nextLine;
                break;
              }
            }
          }
        }

        // If no nationality found with terms, try to find country names
        if (nationality === "Not found") {
          for (const country of countryNames) {
            const countryIndex = lines.findIndex((line) =>
              line.toUpperCase().includes(country)
            );
            if (countryIndex >= 0) {
              nationality = country;
              break;
            }
          }
        }
      }

      setInfo({
        type:
          proofType === "aadhaar"
            ? "Aadhaar"
            : proofType === "pan"
            ? "PAN"
            : "Other",
        name,
        dob,
        number:
          proofType === "other"
            ? nationality
            : proofType === "aadhaar"
            ? nationality
            : nationality,
        passportNumber: proofType === "other" ? passportNumber : "Not found",
        serialNumber: proofType === "other" ? serialNumber : "Not found",
        passportExpiry: proofType === "other" ? passportExpiry : "Not found",
        additionalFields: proofType === "other" ? additionalFields : {},
      });
    } catch (err) {
      console.error("OCR error:", err);
    }
  };

  const capturePhoto = () => {
    if (webcamRef.current) {
      // Use higher quality settings for captured photo
      const imageSrc = webcamRef.current.getScreenshot({
        width: 1280, // Higher resolution
        height: 960, // Higher resolution
        quality: 1.0, // Maximum quality
      });
      setCapturedImage(imageSrc);
    }
  };

  const checkSimilarity = async () => {
    if (!faceImg || !capturedImage) {
      alert(
        "Please ensure both detected face and captured photo are available"
      );
      return;
    }

    setIsComparing(true);
    setSimilarity(null);

    try {
      // Create image elements for both faces
      const detectedFaceImg = new Image();
      const capturedFaceImg = new Image();

      detectedFaceImg.src = faceImg;
      capturedFaceImg.src = capturedImage;

      // Wait for images to load
      await new Promise((resolve) => {
        detectedFaceImg.onload = resolve;
      });
      await new Promise((resolve) => {
        capturedFaceImg.onload = resolve;
      });

      // Detect faces using faceapi
      const detectedFaceDetection = await faceapi
        .detectSingleFace(
          detectedFaceImg,
          new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 })
        )
        .withFaceLandmarks()
        .withFaceDescriptor();

      const capturedFaceDetection = await faceapi
        .detectSingleFace(
          capturedFaceImg,
          new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 })
        )
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!detectedFaceDetection || !capturedFaceDetection) {
        alert("Could not detect faces in one or both images");
        return;
      }

      // Calculate similarity using euclidean distance
      const distance = faceapi.euclideanDistance(
        detectedFaceDetection.descriptor,
        capturedFaceDetection.descriptor
      );
      const similarityPercentage = Math.max(
        0,
        Math.round(100 - distance * 100)
      );

      setSimilarity(similarityPercentage);
    } catch (error) {
      console.error("Error comparing faces:", error);
      alert("Error comparing faces. Please try again.");
    } finally {
      setIsComparing(false);
    }
  };

  // Function to detect faces in real-time from webcam
  const detectFacesInRealTime = async () => {
    if (!webcamRef.current || !webcamRef.current.video) return;

    try {
      const video = webcamRef.current.video;
      const displaySize = {
        width: video.videoWidth,
        height: video.videoHeight,
      };

      // Create a temporary canvas to capture the current frame
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      canvas.width = displaySize.width;
      canvas.height = displaySize.height;

      // Draw the current video frame to canvas
      ctx.drawImage(video, 0, 0, displaySize.width, displaySize.height);

      // Detect faces using faceapi with higher confidence threshold
      const detections = await faceapi
        .detectAllFaces(
          canvas,
          new faceapi.SsdMobilenetv1Options({ minConfidence: 0.8 })
        ) // Even higher confidence
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender(); // Add age and gender detection for human validation

      // Filter detections to ensure they are human faces
      const humanFaceDetections = detections.filter((detection) => {
        // Check if detection has high enough confidence
        if (detection.detection.score < 0.8) return false;

        // Check if face landmarks are detected (indicates human face)
        if (!detection.landmarks || detection.landmarks.positions.length < 68)
          return false;

        // Human face validation using age and gender
        if (!detection.age || !detection.gender) return false;

        // Human faces typically have ages between 0 and 120 years
        if (detection.age < 0 || detection.age > 120) return false;

        // Check for typical human face proportions
        const box = detection.detection.box;
        const aspectRatio = box.width / box.height;

        // Human faces typically have aspect ratio between 0.8 and 1.2 (more strict)
        if (aspectRatio < 0.8 || aspectRatio > 1.2) return false;

        // Check face size relative to image (humans are typically a reasonable size)
        const faceArea = box.width * box.height;
        const imageArea = displaySize.width * displaySize.height;
        const faceRatio = faceArea / imageArea;

        // Face should be between 5% and 50% of the image area
        if (faceRatio < 0.05 || faceRatio > 0.5) return false;

        // Additional validation: check for typical human face distance features
        if (detection.landmarks) {
          const positions = detection.landmarks.positions;

          // Check if we have enough key points for human face validation
          if (positions.length < 68) return false;

          // Check for typical human face structure (eyes, nose, mouth positioning)
          // Get approximate positions of key facial features
          const leftEye = positions.find((p) => p.name === "leftEye");
          const rightEye = positions.find((p) => p.name === "rightEye");
          const nose = positions.find((p) => p.name === "nose");
          const mouth = positions.find((p) => p.name === "mouth");

          // If we have these key points, validate their relative positions
          if (leftEye && rightEye && nose && mouth) {
            // Calculate eye distance and face height
            const eyeDistance = Math.abs(rightEye.x - leftEye.x);
            const faceHeight = Math.abs(
              mouth.y - Math.min(leftEye.y, rightEye.y)
            );

            // Human faces typically have eye-to-face height ratio between 0.3 and 0.5
            const eyeToFaceHeightRatio = eyeDistance / faceHeight;
            if (eyeToFaceHeightRatio < 0.3 || eyeToFaceHeightRatio > 0.5)
              return false;
          }
        }

        return true;
      });

      setDetectedFaces(humanFaceDetections);
    } catch (error) {
      console.error("Error detecting faces:", error);
    }
  };

  // Effect to start/stop face detection when webcam is active
  useEffect(() => {
    let detectionInterval;

    if (webcamRef.current && webcamRef.current.video) {
      setIsDetectingFaces(true);
      detectionInterval = setInterval(detectFacesInRealTime, 100); // Detect every 100ms
    }

    return () => {
      if (detectionInterval) {
        clearInterval(detectionInterval);
      }
      setIsDetectingFaces(false);
      setDetectedFaces([]);
    };
  }, [webcamRef.current?.video]);

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col lg:flex-row">
      <main className="flex-1 bg-white p-4 md:p-6 shadow-md">
        <div className="border-t-2 border-b-2 border-gray-400 p-4">
          <h2 className="text-xl md:text-2xl font-bold mb-4 text-center">
            Proof of Identity (POI)
          </h2>

          <fieldset className="border border-gray-500 rounded px-2 py-1 mb-4">
            <legend className="px-1 text-gray-700 font-semibold text-sm">
              Proof of Identity Type
            </legend>
            <select
              value={proofType}
              onChange={(e) => setProofType(e.target.value)}
              className="w-full p-1 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
            >
              <option value="">-- Select Proof Type --</option>
              <option value="aadhaar">Aadhaar Card</option>
              <option value="pan">PAN Card</option>
              <option value="other">Other</option>
            </select>
          </fieldset>

          {/* Upload Front */}
          <div className="mb-4">
            <label
              htmlFor="frontUpload"
              className="flex flex-col md:flex-row items-center gap-2 md:gap-4 p-4 border-2 border-dashed border-gray-500 rounded cursor-pointer hover:bg-gray-50"
            >
              <div className="border border-2 border-gray-500 p-2 rounded-2xl flex-shrink-0">
                <img
                  src={UploadIcon}
                  alt="Upload Icon"
                  className="h-12 w-16 md:h-20 md:w-30"
                />
              </div>
              <div className="text-center md:text-left md:pl-3 flex-1">
                <span className="text-gray-700 font-medium text-sm md:text-base">
                  <span className="text-blue-500">Upload</span>{" "}
                  {proofType === "aadhaar"
                    ? "Aadhaar"
                    : proofType === "pan"
                    ? "PAN"
                    : "Other"}{" "}
                  front
                  <div className="text-xs md:text-sm">
                    Only .jpg, .pdf, .png, max size 5MB
                  </div>
                </span>
              </div>
            </label>
            <input
              id="frontUpload"
              type="file"
              accept="image/*"
              onChange={handleFrontUpload}
              className="hidden"
            />
          </div>

          {/* Upload Back */}
          <div className="mb-4">
            <label
              htmlFor="backUpload"
              className="flex flex-col md:flex-row items-center gap-2 md:gap-4 p-4 border-2 border-dashed border-gray-500 rounded cursor-pointer hover:bg-gray-50"
            >
              <div className="border border-2 border-gray-500 p-2 rounded-2xl flex-shrink-0">
                <img
                  src={UploadIcon}
                  alt="Upload Icon"
                  className="h-12 w-16 md:h-20 md:w-30"
                />
              </div>
              <div className="text-center md:text-left md:pl-3 flex-1">
                <span className="text-gray-700 font-medium text-sm md:text-base">
                  <span className="text-blue-500">Upload</span>{" "}
                  {proofType === "aadhaar"
                    ? "Aadhaar"
                    : proofType === "pan"
                    ? "PAN"
                    : "Other"}{" "}
                  back
                  <div className="text-xs md:text-sm">
                    Only .jpg, .pdf, .png, max size 5MB
                  </div>
                </span>
              </div>
            </label>
            <input
              id="backUpload"
              type="file"
              accept="image/*"
              onChange={handleBackUpload}
              className="hidden"
            />
          </div>
        </div>

        {/* Output */}
        <div className="pt-4">
          <h3 className="text-lg font-semibold mb-3 text-center">
            Uploaded Previews
          </h3>
          <div className="flex flex-col lg:flex-row gap-4 lg:gap-6 justify-center">
            {frontImg && (
              <div className="flex flex-col items-center">
                <h4 className="text-sm font-medium mb-1 text-gray-600 text-center">
                  Front Side
                </h4>
                <img
                  src={frontImg}
                  alt="Front Preview"
                  className="w-40 h-32 sm:w-48 sm:h-36 md:w-56 md:h-44 lg:w-60 lg:h-40 object-cover rounded border"
                />
              </div>
            )}
            {backImg && (
              <div className="flex flex-col items-center">
                <h4 className="text-sm font-medium mb-1 text-gray-600 text-center">
                  Back Side
                </h4>
                <img
                  src={backImg}
                  alt="Back Preview"
                  className="w-40 h-32 sm:w-48 sm:h-36 md:w-56 md:h-44 lg:w-60 lg:h-40 object-cover rounded border"
                />
              </div>
            )}
            {faceImg && (
              <div className="flex flex-col lg:flex-row gap-4 lg:gap-6 items-center">
                <div className="flex flex-col items-center">
                  <h4 className="text-sm font-medium mb-1 text-gray-600 text-center">
                    Detected Face
                  </h4>
                  <img
                    src={faceImg}
                    alt="Detected Face"
                    className="w-28 h-28 sm:w-32 sm:h-32 md:w-36 md:h-36 lg:w-40 lg:h-40 object-cover rounded-full border-2 border-green-500"
                  />
                </div>
                {capturedImage && (
                  <div className="flex flex-col items-center">
                    <h4 className="text-sm font-medium mb-1 text-gray-600 text-center">
                      Captured Photo
                    </h4>
                    <img
                      src={capturedImage}
                      alt="Captured Photo"
                      className="w-28 h-28 sm:w-32 sm:h-32 md:w-36 md:h-36 lg:w-40 lg:h-40 rounded-full border-2 border-purple-500"
                    />
                    <button
                      onClick={checkSimilarity}
                      disabled={isComparing}
                      className={`mt-2 w-20 md:w-24 px-3 py-2 rounded-lg font-medium transition-colors text-xs md:text-sm ${
                        isComparing
                          ? "bg-gray-400 cursor-not-allowed"
                          : "bg-green-500 hover:bg-green-600 text-white"
                      }`}
                    >
                      {isComparing ? "Comparing..." : "Check Similarity"}
                    </button>
                    {similarity !== null && (
                      <div className="mt-2 p-2 rounded-lg border w-20 md:w-24 text-center">
                        <p className="text-sm md:text-lg font-bold text-green-600">
                          {similarity}%
                        </p>
                      </div>
                    )}
                  </div>
                )}
                <div className="flex flex-col items-center">
                  <h4 className="text-sm font-medium mb-1 text-gray-600 text-center">
                    Webcam
                  </h4>
                  <div className="relative w-28 h-28 sm:w-32 sm:h-32 md:w-36 md:h-36 lg:w-40 lg:h-40">
                    <Webcam
                      ref={webcamRef}
                      audio={false}
                      screenshotFormat="image/jpeg"
                      videoConstraints={{
                        facingMode: facingMode,
                        width: { ideal: 1280 }, // Higher resolution for better quality
                        height: { ideal: 960 }, // Higher resolution for better quality
                        facingMode: facingMode,
                      }}
                      className="w-full h-full object-cover rounded-full border-2 border-blue-500"
                    />
                    {/* Green line box around detected faces */}
                    {detectedFaces.length > 0 && (
                      <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-full">
                        {detectedFaces.map((detection, index) => {
                          const box = detection.detection.box;
                          const webcamSize =
                            window.innerWidth < 768
                              ? 112
                              : window.innerWidth < 1024
                              ? 128
                              : 160; // w-28=112, w-32=128, w-36=144, w-40=160
                          const scale = webcamSize / 640; // Scale factor based on actual webcam size

                          // Ensure the box stays within webcam boundaries
                          const left = Math.max(
                            0,
                            Math.min(
                              box.x * scale,
                              webcamSize - box.width * scale
                            )
                          );
                          const top = Math.max(
                            0,
                            Math.min(
                              box.y * scale,
                              webcamSize - box.height * scale
                            )
                          );
                          const width = Math.min(
                            box.width * scale,
                            webcamSize - left
                          );
                          const height = Math.min(
                            box.height * scale,
                            webcamSize - top
                          );

                          return (
                            <div
                              key={index}
                              className="absolute border-2 border-green-500 rounded"
                              style={{
                                left: `${left}px`,
                                top: `${top}px`,
                                width: `${width}px`,
                                height: `${height}px`,
                              }}
                            />
                          );
                        })}
                      </div>
                    )}
                  </div>
                  <div className="flex flex-col gap-2 mt-2 w-full">
                    <button
                      onClick={() =>
                        setFacingMode(
                          facingMode === "user" ? "environment" : "user"
                        )
                      }
                      className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors text-xs md:text-sm w-full"
                    >
                      Rotate Camera
                    </button>
                    <button
                      onClick={capturePhoto}
                      disabled={detectedFaces.length === 0}
                      className={`px-3 py-1 rounded transition-colors text-xs md:text-sm w-full ${
                        detectedFaces.length === 0
                          ? "bg-gray-400 cursor-not-allowed"
                          : "bg-blue-500 hover:bg-blue-600 text-white"
                      }`}
                    >
                      Click Photo
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Processing indicator */}
          {loading && (
            <p className="text-blue-500 mt-2 text-center">Processing...</p>
          )}

          {info && (
            <div className="mt-6 bg-gray-50 p-4 rounded border w-full md:w-fit mx-auto">
              <h4 className="text-lg font-semibold mb-2 text-gray-700 text-center">
                Extracted {info.type} Info
              </h4>
              <div className="space-y-1 text-sm">
                <p>
                  <strong>Name:</strong> {info.name}
                </p>
                <p>
                  <strong>DOB:</strong> {info.dob}
                </p>
                <p>
                  <strong>
                    {info.type === "Aadhaar"
                      ? "Aadhaar Number"
                      : info.type === "PAN"
                      ? "PAN Number"
                      : "Nationality"}
                  </strong>{" "}
                  {info.number}
                </p>
                {info.type === "Other" &&
                  info.passportNumber !== "Not found" && (
                    <p>
                      <strong>Passport Number:</strong> {info.passportNumber}
                    </p>
                  )}
                {info.type === "Other" && info.serialNumber !== "Not found" && (
                  <p>
                    <strong>Serial Number:</strong> {info.serialNumber}
                  </p>
                )}
                {info.type === "Other" &&
                  info.passportExpiry !== "Not found" && (
                    <p>
                      <strong>Passport Expiry:</strong> {info.passportExpiry}
                    </p>
                  )}
                {info.type === "Other" &&
                  Object.keys(info.additionalFields).length > 0 &&
                  Object.entries(info.additionalFields).map(([key, value]) => (
                    <p key={key}>
                      <strong>
                        {key.charAt(0).toUpperCase() +
                          key.slice(1).replace(/([A-Z])/g, " $1")}
                        :
                      </strong>{" "}
                      {value}
                    </p>
                  ))}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Sidebar - Only visible on large screens and above */}
      <aside className="hidden lg:block lg:w-80 p-6 flex justify-center">
        <h2 className="text-xl font-bold mb-4">Sidebar</h2>
      </aside>
    </div>
  );
}
