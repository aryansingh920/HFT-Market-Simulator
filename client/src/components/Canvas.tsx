// src/components/Canvas.tsx

"use client";

import React, { useState } from "react";
import dynamic from "next/dynamic";
import Navbar from "@/components/Navbar";
import OrderBook from "@/components/OrderBook";
import TradeFeed from "@/components/TradeFeed";

// Dynamically import ResizableDraggable with SSR disabled
const ResizableDraggable = dynamic(() => import("./ResizableDraggable"), {
  ssr: false,
});

interface ComponentConfig {
  id: string;
  type: string;
  x: number;
  y: number;
}

const Canvas: React.FC = () => {
  const [components, setComponents] = useState<ComponentConfig[]>([]);

  // Handler to add new components
  const handleAddComponent = (type: string) => {
    const newComponent: ComponentConfig = {
      id: `${type}-${Date.now()}`,
      type,
      x: 100, // Default position
      y: 100,
    };
    setComponents((prev) => [...prev, newComponent]);
  };

  // Render components based on type
  const renderComponent = (type: string) => {
    switch (type) {
      case "OrderBook":
        return <OrderBook />;
      case "TradeFeed":
        return <TradeFeed />;
      default:
        return null;
    }
  };

  return (
    <div className="w-full h-screen bg-gray-200 relative">
      {/* Navbar */}
      <Navbar onAddComponent={handleAddComponent} />

      {/* Canvas */}
      <div className="relative w-full h-full">
        {components.map((component) => (
          <ResizableDraggable
            key={component.id}
            defaultPosition={{ x: component.x, y: component.y }}
          >
            {renderComponent(component.type)}
          </ResizableDraggable>
        ))}
      </div>
    </div>
  );
};

export default Canvas;
