module InlineFixes

import util::IDEServices;
import util::Math;
import List;
import String;
import Store;

list[CodeAction] prepareNoDeploymentFixes(loc src)
    = [
        action(title="Remove monitoring step", edits=[changed(src.top, [replace(src, "")])]),
        action(title="Add deployment step before monitoring", edits=[changed(src.top, [replace(insertionPointBefore(src), deploymentSnippet(src))])])
      ];

list[CodeAction] prepareNoConnectionFixes(loc src)
    = [
        action(title="Remove monitoring step", edits=[changed(src.top, [replace(src, "")])])
      ];

loc insertionPointBefore(loc src) = src[length=0][end=src.begin];

str deploymentSnippet(loc src) {
    str ind = intercalate("", [" " | _ <- [0..src.begin.column]]);
    str innerInd = ind + "    ";
    return "deployment (\n"
         + innerInd + "port = 8000\n"
         + ind + ")\n"
         + "\n"
         + ind;
}

loc removeEntryLoc(list[loc] \all, int idx) {
    loc item = \all[idx];
    if (size(\all) == 1) {
        return item;
    }
    if (idx == size(\all) - 1) {
        loc prev = \all[idx - 1];
        int startOffset = prev.offset + prev.length;
        int endOffset = item.offset + item.length;
        return item.top[offset=startOffset][length=endOffset-startOffset][begin=prev.end][end=item.end];
    }
    loc next = \all[idx + 1];
    int startOffset = item.offset;
    int endOffset = next.offset;
    return item.top[offset=startOffset][length=endOffset-startOffset][begin=item.begin][end=next.begin];
}

list[CodeAction] removeEntryFix(str title, list[loc] \all, int idx)
    = [action(title=title, edits=[changed(\all[idx].top, [replace(removeEntryLoc(\all, idx), "")])])];

loc paramNameLoc(loc paramSrc, str name) {
    int len = size(name);
    return paramSrc[length=len][end=<paramSrc.begin.line, paramSrc.begin.column + len>];
}

list[CodeAction] invalidParamNameFixes(loc paramSrc, str invalidName, ParamSignature sig) {
    loc target = paramNameLoc(paramSrc, invalidName);
    list[str] validNames = [n | n <- sig];
    return [
        action(title="Replace with \'<validName>\'", edits=[changed(target.top, [replace(target, validName)])])
        | str validName <- validNames
    ];
}

int levenshtein(str a, str b) {
    list[int] la = chars(a);
    list[int] lb = chars(b);
    int m = size(la);
    int n = size(lb);
    list[list[int]] d = [[0 | _ <- [0..n+1]] | _ <- [0..m+1]];

    for (int i <- [0..m+1]) {
        d[i][0] = i;
    }
    for (int j <- [0..n+1]) {
        d[0][j] = j;
    }

    for (int i <- [1..m+1]) {
        for (int j <- [1..n+1]) {
            int cost = (la[i-1] == lb[j-1]) ? 0 : 1;
            d[i][j] = min([
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            ]);
        }
    }
    return d[m][n];
}

list[CodeAction] schemaSuggestionFixes(loc src, str invalidName, Schema schema) {
    list[str] names = [n | n <- schema];
    lrel[int dist, str name] ranked = sort([<levenshtein(invalidName, n), n> | n <- names]);
    list[str] topNames = [name | <int _, str name> <- ranked][0..min(5, size(ranked))];
    return [
        action(title="Replace with \'<n>\'", edits=[changed(src.top, [replace(src, "\"<n>\"")])])
        | str n <- topNames
    ];
}