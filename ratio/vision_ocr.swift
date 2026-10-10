// On-device OCR for scanned PDFs with Apple's Vision framework (no network, no extra install).
// Usage: ratio-ocr <file.pdf>  ->  {"pages": ["page 1 text", ...]} on stdout. Compiled by ratio/ocr.py.
import CoreGraphics
import Foundation
import Vision

let scale: CGFloat = 3.0  // about 216 dpi for a letter-size page

func render(_ page: CGPDFPage) -> CGImage? {
    let box = page.getBoxRect(.mediaBox)
    let width = Int(box.width * scale), height = Int(box.height * scale)
    guard width > 0, height > 0, let context = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }
    context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.scaleBy(x: scale, y: scale)
    context.translateBy(x: -box.origin.x, y: -box.origin.y)
    context.drawPDFPage(page)
    return context.makeImage()
}

func recognise(_ image: CGImage) -> String {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = ["en-US"]
    try? VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    let lines = (request.results ?? []).sorted {
        let a = $0.boundingBox, b = $1.boundingBox
        return abs(a.maxY - b.maxY) > 0.01 ? a.maxY > b.maxY : a.minX < b.minX  // top to bottom, then left to right
    }
    return lines.compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
}

guard CommandLine.arguments.count == 2,
      let document = CGPDFDocument(URL(fileURLWithPath: CommandLine.arguments[1]) as CFURL) else {
    FileHandle.standardError.write("usage: ratio-ocr <file.pdf>\n".data(using: .utf8)!)
    exit(2)
}
var pages: [String] = []
if document.numberOfPages > 0 {
    for number in 1...document.numberOfPages {
        if let page = document.page(at: number), let image = render(page) {
            pages.append(recognise(image))
        } else {
            pages.append("")
        }
    }
}
FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: ["pages": pages]))
