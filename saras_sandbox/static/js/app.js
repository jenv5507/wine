// from table_to_filter.js
const tableData = data;
console.log("Table data assigned");

// get table references
var tbody = d3.select("tbody");


function buildTable(data) {
  // console.log(data);
  //first clear data in the table
  tbody.html("");

  //loop through elements in the array
  data.slice(0,50).forEach((dataRow) => {
      //create a variable that will append a row to the table body ("tr" stands for table row in HTML)
      let row = tbody.append("tr");
      //loop through each field in the row ("td" stads for table data in HTML)
      Object.values(dataRow).forEach((val) => {
          //set up action of appending data into a table data tag, in order to put each value of the sighting into a cell
          let cell = row.append("td");
          //add the values to the cell
          cell.text(val);
          }
      );
  });
};


// Create a variable to keep track of all the filters as an object.
var filters = {};

// Use this function to update the filters. 
function updateFilters() {
    // Save the element that was changed as a variable.
    let inputElement = d3.select(this)
    // Save the id that was changed as a variable.
    let inputId = inputElement.attr("id");
    // Save the value of the filter that was changed as a variable.
    let inputValue = inputElement.property("value");
      // If a filter value was entered then add that filterId and value
    // to the filters list. Otherwise, clear that filter from the filters object.
    if (inputValue) {
      filters[inputId] = inputValue;
    }
    else {
      delete filters[inputId];
    };
    // Call function to apply all filters and rebuild the table
    filterTable();
  };
  
  // Use this function to filter the table when data is entered.
  function filterTable() {
    console.log("filtering");
    // Set the filtered data to the tableData.
    let filteredData = tableData; 
    console.log(filteredData);
    // Loop through all of the filters and keep any data that
    // matches the filter values
    // Object.entries(filters).forEach(([key, value]) => {
    //   console.log(key);
    //   filteredData = filteredData.slice(0,50).filter(row => row[key] === value);
    //   console.log(filteredData)
    // });
    Object.entries(filters).forEach(([key,value]) => {
      console.log(key);
      filteredData = filteredData.filter(row => row[key] === value);
      console.log(filteredData);
      console.log(value);
    });
    // Finally, rebuild the table using the filtered data
    buildTable(filteredData);
    console.log(filteredData);    
  };
  
  // Attach an event to listen for changes to each filter
  d3.selectAll("input").on("change", updateFilters);
  
  // Build the table when the page loads
  buildTable(tableData);
