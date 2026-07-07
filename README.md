# 🛡️ VeritasEdge: Decentralized AI Proctoring Suite

![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00A971?style=for-the-badge&logo=google&logoColor=white)

VeritasEdge is a lightweight, zero-backend Edge AI Proctoring Proof of Concept (POC). It transitions heavy, server-dependent computer vision tasks completely to the client side. By running directly in the browser via WebAssembly (WASM), it ensures **$0 server compute costs** and **absolute user privacy** (video frames never leave the candidate's device).

## ✨ Key Features

* **Zero-Backend Architecture:** 100% client-side inference using Google's `@mediapipe/tasks-vision`.
* **Gaze Tracking & 5-Second Rule:** Real-time iris tracking calculates pitch/yaw. If a candidate looks away from the screen for > 5 continuous seconds, a critical security flag is triggered.
* **Multi-Face Security Alerts:** Instantaneous detection of multiple individuals within the webcam frame to prevent unauthorized assistance.
* **Anti-Jitter Telemetry (EMA):** Raw vision telemetry is smoothed using an Exponential Moving Average (EMA) filter, providing a stable, flicker-free UI dashboard.
* **Custom OGL WebGL Landing Page:** A highly optimized, custom `ogl` WebGL shader background that avoids the standard tree-shaking and memory leak issues of heavier 3D React wrapper libraries.

## 🛠️ Tech Stack

* **Core:** React 18, TypeScript, Vite
* **Computer Vision:** Google MediaPipe (`FaceLandmarker`)
* **Graphics/Landing Page:** OGL (Ultra-lightweight WebGL)
* **Styling:** Modern CSS / Flexbox

## 🚀 Getting Started

### ⚙️ Prerequisites
Make sure you have Node.js installed on your local machine.

🚀 Installation
Clone the repository:

```Bash
git clone https://github.com/shresthgoel77-png/AI-proctoring-System.git
```

Navigate into the project directory:

```Bash
cd AI-proctoring-System
```
Install dependencies: (This will install all required packages, including Google MediaPipe and OGL)
```Bash
npm install
```
Start the local development server:
```Bash
npm run dev
```
🌍 Deployment

This project is fully optimized for static deployment. You can deploy the dist folder directly to Vercel, Netlify, or GitHub Pages with zero backend configuration required.

To generate the optimized production build, run:
```Bash
npm run build
```
---
Built by Shredev with ❤️
---
