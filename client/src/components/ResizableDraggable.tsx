import React, { ReactNode } from "react";
import { Rnd } from "react-rnd";

interface ResizableDraggableProps {
  children: ReactNode;
  defaultPosition?: { x: number; y: number };
  defaultSize?: { width: number; height: number };
}

const ResizableDraggable: React.FC<ResizableDraggableProps> = ({
  children,
  defaultPosition = { x: 0, y: 0 },
  defaultSize = { width: 300, height: 200 },
}) => {
  return (
    <Rnd
      default={{
        x: defaultPosition.x,
        y: defaultPosition.y,
        width: defaultSize.width,
        height: defaultSize.height,
      }}
      bounds="parent"
      className="bg-white shadow-md rounded-lg border border-gray-300"
    >
      {children}
    </Rnd>
  );
};

export default ResizableDraggable;
