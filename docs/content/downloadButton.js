import React, { useState } from 'react';
import sendAllTabs from './sendAllTabs';

const DownloadButton = ({ filename }) => {
    if (!filename) {
        console.error("Filename is not provided or invalid.");
        return null; // Optionally handle this case appropriately
    }
    
    const [isHovered, setIsHovered] = useState(false);

    const baseStyle = {
        padding: '10px',
        borderRadius: '10px',
        border: '0',
        color: '#000',
        backgroundColor: '#C4F800',
        fontWeight: 'bold',
        cursor: 'pointer', // Always show pointer on hover
    };

    const hoverStyle = {
        backgroundColor: '#B0E000' // Slightly darker when hovered
    };

    return (
        <button
            style={isHovered ? { ...baseStyle, ...hoverStyle } : baseStyle}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            onClick={() => sendAllTabs(filename)}
        >
            Generate notebook from all selected tabs
        </button>
    );
};

export default DownloadButton;
