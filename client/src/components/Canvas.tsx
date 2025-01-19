"use client";
import React from "react";
import dynamic from "next/dynamic";
import OrderBook from "./OrderBook";

// Dynamically import ResizableDraggable with SSR disabled
const ResizableDraggable = dynamic(() => import("./ResizableDraggable"), {
  ssr: false,
  loading: () => <div>Loading...</div>,
});

const Canvas = () => {
  return (
    <div className="w-full h-screen bg-gray-200 relative">
      <ResizableDraggable defaultPosition={{ x: 50, y: 50 }}>
        <OrderBook />
      </ResizableDraggable>
    </div>
  );
};

export default Canvas;
