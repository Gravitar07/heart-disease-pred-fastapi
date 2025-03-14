/**
 * Converts LLM report text to markdown format
 * @param {string} reportText - The original report text
 * @returns {string} Markdown formatted text
 */
function convertToMarkdown(reportText) {
  if (!reportText) return '';
  
  // Split the text into lines
  let lines = reportText.split('\n');
  let markdownLines = [];
  
  for (let line of lines) {
    // Check if the line is already a markdown heading (starts with # symbols)
    if (line.trim().match(/^#{1,6}\s/)) {
      // Line is already a proper markdown heading, keep it as is
      markdownLines.push(line);
    }
    // Convert section headers (lines starting with ** and ending with **)
    else if (line.trim().startsWith('**') && line.trim().endsWith('**')) {
      // Remove the asterisks
      let heading = line.trim().replace(/^\*\*|\*\*$/g, '');
      
      // Default to h2 for all converted headings for consistency
      markdownLines.push(`## ${heading}`);
    } 
    // Convert list items (lines starting with *)
    else if (line.trim().startsWith('* ')) {
      markdownLines.push(line); // Already in markdown format
    }
    // Handle bold text
    else if (line.includes('**')) {
      markdownLines.push(line); // Keep the original markdown bold format
    } 
    // Regular text
    else if (line.trim()) {
      markdownLines.push(line);
    }
    // Preserve empty lines
    else {
      markdownLines.push('');
    }
  }
  
  return markdownLines.join('\n');
} 