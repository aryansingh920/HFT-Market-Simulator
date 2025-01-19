/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState, useEffect } from "react";

export const usePersistentLayout = (key: string, defaultValue: any) => {
  const [layout, setLayout] = useState(() => {
    const saved = localStorage.getItem(key);
    return saved ? JSON.parse(saved) : defaultValue;
  });

  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(layout));
  }, [key, layout]);

  return [layout, setLayout];
};
