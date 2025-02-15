import React from "react";

interface NavbarProps {
  onAddComponent: (type: string) => void;
}

const Navbar: React.FC<NavbarProps> = ({ onAddComponent }) => {
  return (
    <div className="w-full bg-blue-600 text-white flex items-center p-4 shadow-lg">
      <h1 className="text-lg font-bold mr-4">Simulator Dashboard</h1>
      <button
        onClick={() => onAddComponent("OrderBook")}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mr-2"
      >
        Add Order Book
      </button>
      <button
        onClick={() => onAddComponent("TradeFeed")}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mr-2"
      >
        Add Trade Feed
      </button>
      <button
        onClick={() => onAddComponent("MarketChart")}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
      >
        Add Market Chart
      </button>
    </div>
  );
};

export default Navbar;
