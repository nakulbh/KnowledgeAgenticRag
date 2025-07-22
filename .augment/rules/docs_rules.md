---
type: "manual"
---

# Code Documentation Rules for AI Assistant

## Core Instructions
When asked to document code, ALWAYS follow this comprehensive structure. Generate thorough, detailed documentation that goes beyond surface-level explanations. Focus on the WHY, HOW, and WHAT-IF scenarios, not just WHAT the code does.

---

## Required Documentation Structure

### 1. 📛 Title
- Format: `[Action Verb] [Object] [Method/Context]`
- Make it descriptive and searchable
- Examples: "Normalize Time Series Using Min-Max Scaling", "Generate JWT with Custom Claims"

### 2. 🔍 High-Level Overview
Write 2-3 sentences covering:
- **What** it does in business/functional terms
- **Why** it's useful or necessary
- **When/Where** you'd use it
- **Value proposition** - why this approach

### 3. 🔧 Technical Deep Dive
Break down the internal mechanics:
- Step-by-step algorithm explanation
- Key transformations and calculations
- Control flow and decision logic
- Implementation choices and trade-offs
- Why specific techniques were selected over alternatives

### 4. 🔢 Parameters & Returns
For each parameter/return value, include:
- **Type annotations** (be specific: `List[Dict[str, int]]` not just `list`)
- **Purpose** and expected content
- **Constraints** (valid ranges, formats, requirements)
- **Default values** and their rationale
- **Optional vs required** status
- **Examples** of valid inputs

Format:
```
Parameters:
├── param_name (Type)
│   ├── Purpose: What this parameter does
│   ├── Constraints: Valid values/ranges
│   ├── Default: value (if applicable)
│   └── Example: sample_value

Returns:
└── return_type
    ├── Description: What gets returned
    └── Structure: Format/schema details
```

### 5. 💡 Implementation Details
Explain the "how" behind key decisions:
- **Algorithms** used and why
- **Data structures** chosen
- **Performance optimizations** applied
- **Design patterns** implemented
- **Libraries/dependencies** leveraged

### 6. 🚀 Usage Examples
Provide 2-3 practical examples:
- **Basic usage** - simplest case
- **Advanced usage** - with optional parameters
- **Real-world scenario** - actual use case context
- Include **expected outputs** for each example

### 7. ⚠️ Edge Cases & Error Handling
Document what can go wrong:
- **Input validation** and error responses
- **Boundary conditions** (empty inputs, null values)
- **Performance limits** (memory, time constraints)
- **Exception types** that might be raised
- **Graceful degradation** strategies

### 8. 📊 Performance & Complexity
Include technical analysis:
- **Time complexity**: Big O notation with explanation
- **Space complexity**: Memory usage patterns
- **Scalability considerations**: How it performs with large data
- **Bottlenecks**: Known performance limitations
- **Optimization opportunities**: Where improvements could be made

### 9. 🔗 Dependencies & Requirements
List everything needed:
- **External libraries** with version requirements
- **System requirements** (OS, Python version, etc.)
- **Environment setup** if complex
- **Optional dependencies** and their benefits

### 10. 🌐 Integration & Context
Show how it fits in the bigger picture:
- **Related functions** in the codebase
- **Typical workflow** where this is used
- **Data pipeline** position
- **API endpoints** that might call it
- **Database interactions** if any

### 11. 🔮 Future Improvements
Suggest enhancement opportunities:
- **Performance optimizations** possible
- **Feature additions** that make sense
- **Refactoring opportunities** for better maintainability
- **Integration possibilities** with other systems

---

## Documentation Quality Standards

### Writing Style
- Use **active voice** and **present tense**
- Write for a **technical audience** but explain complex concepts clearly
- Include **code snippets** liberally with proper syntax highlighting
- Use **emoji headers** for visual organization and quick scanning
- Balance **technical depth** with **readability**

### Technical Depth
- Go beyond obvious functionality
- Explain **implementation rationale**
- Include **performance implications**
- Address **security considerations** when relevant
- Mention **testing strategies** if applicable

### Completeness Checklist
- [ ] Clear, searchable title
- [ ] Business context and value proposition
- [ ] Step-by-step technical explanation
- [ ] Complete parameter specifications with examples
- [ ] Implementation details and design choices
- [ ] Multiple usage examples with outputs
- [ ] Comprehensive error handling documentation
- [ ] Performance analysis and complexity
- [ ] All dependencies and requirements listed
- [ ] Integration context provided
- [ ] Future improvement suggestions included

---

## Example Quality Indicators

### ❌ Avoid Surface-Level Documentation:
"This function sorts a list of numbers in ascending order."

### ✅ Provide Deep, Valuable Documentation:
"This function implements an optimized quicksort algorithm with median-of-three partitioning to sort numerical data. It's designed for financial time series where maintaining stable sorting of equal elements is crucial for temporal ordering. The implementation switches to insertion sort for small arrays (n < 10) to avoid quicksort's overhead on small datasets, achieving O(n log n) average performance while handling worst-case O(n²) scenarios gracefully through randomized pivot selection."

---

## Final Reminder
Every documentation request should result in comprehensive, production-ready documentation that serves as both a reference guide and educational resource. Focus on creating documentation that helps developers understand not just HOW to use the code, but WHY it exists and WHEN to use it.